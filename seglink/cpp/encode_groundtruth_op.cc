#define EIGEN_USE_THREADS

#include <cmath>
#include <climits>
#include <cassert>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

#include "utilities.h"

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;


REGISTER_OP("EncodeGroundtruth")
    .Attr("anchor_size: float")
    .Attr("pos_scale_diff_thresh: float = 1.5")
    .Attr("neg_scale_diff_thresh: float = 2.0")
    .Attr("cross_links: bool = false")
    .Input("gt_rboxes: float32")
    .Input("gt_counts: int32")
    .Input("map_size: int32")
    .Input("image_size: int32")
    .Input("match_status_below: int32")
    .Input("match_indices_below: int32")
    .Output("match_status: int32")
    .Output("link_status: int32")
    .Output("gt_offsets: float32")
    .Output("match_indices: int32");


template <typename Device, typename T>
class EncodeGroundtruthOp : public OpKernel {
public:
  explicit EncodeGroundtruthOp(OpKernelConstruction* context)
    : OpKernel(context),
      rbox_dim_(5),
      offsets_dim_(6),
      n_within_links_(8),
      n_cross_links_(4),
      cross_stride_(2) {
    OP_REQUIRES_OK(context, context->GetAttr("anchor_size", &anchor_size_));
    OP_REQUIRES(context, anchor_size_ > 0,
                errors::InvalidArgument("Expected anchor_size > 0, got ", anchor_size_));

    OP_REQUIRES_OK(context, context->GetAttr("pos_scale_diff_thresh", &pos_scale_diff_thresh_));
    OP_REQUIRES(context, pos_scale_diff_thresh_ >= 1,
                errors::InvalidArgument("Expected pos_scale_diff_thresh >= 1, got ", pos_scale_diff_thresh_));

    OP_REQUIRES_OK(context, context->GetAttr("neg_scale_diff_thresh", &neg_scale_diff_thresh_));
    OP_REQUIRES(context, neg_scale_diff_thresh_ >= 1 && neg_scale_diff_thresh_ > pos_scale_diff_thresh_,
                errors::InvalidArgument("Expected neg_scale_diff_thresh >= 1 and greater than pos_scale_diff_thresh, got: ",
                                        neg_scale_diff_thresh_));

    OP_REQUIRES_OK(context, context->GetAttr("cross_links", &cross_links_));

    n_links_ = cross_links_ ? n_within_links_ + n_cross_links_ : n_within_links_;
  }

  void Compute(OpKernelContext* context) override {
    // read input
    const Tensor& gt_rboxes = context->input(0);
    const Tensor& gt_counts = context->input(1);
    const Tensor& map_size = context->input(2);
    const Tensor& image_size = context->input(3);
    const Tensor& match_status_below = context->input(4);
    const Tensor& match_indices_below = context->input(5);
    OP_REQUIRES(context, gt_rboxes.dims() == 3 && gt_rboxes.dim_size(2) == 5,
                errors::InvalidArgument("Expected shape of gt_rboxes is [*,*,5], got ",
                                        gt_rboxes.shape().DebugString()));
    OP_REQUIRES(context, gt_counts.dims() == 1,
                errors::InvalidArgument("Expected shape of gt_counts is [*], got ",
                                        gt_counts.shape().DebugString()));
    OP_REQUIRES(context, map_size.dims() == 1 && map_size.dim_size(0) == 3,
                errors::InvalidArgument("Expected shape of map_size is [3], got ",
                                        map_size.shape().DebugString()));
    OP_REQUIRES(context, image_size.dims() == 1 && image_size.dim_size(0) == 3,
                errors::InvalidArgument("Expected image_size has shape [3], got ",
                                        image_size.shape().DebugString()));
    OP_REQUIRES(context, match_status_below.dims() == 3,
                errors::InvalidArgument("Expected shape of match_status_below is [*,*,*], got ",
                                        match_status_below.shape().DebugString()));
    OP_REQUIRES(context, match_indices_below.dims() == 3,
                errors::InvalidArgument("Expected shape of match_indices_below is [*,*,*], got ",
                                        match_indices_below.shape().DebugString()));

    // allocate output
    const int batch_size = gt_rboxes.dim_size(0);
    auto map_size_tensor = map_size.vec<int32>();
    const int map_h = map_size_tensor(0);
    const int map_w = map_size_tensor(1);

    // output 0, [batch, map_h, map_w], match status
    Tensor* match_status = nullptr;
    OP_REQUIRES_OK(
      context, context->allocate_output(0, {batch_size, map_h, map_w}, &match_status));
    // output 1, [batch, map_h, map_w, 8], link status
    Tensor* link_status = nullptr;
    OP_REQUIRES_OK(context,
      context->allocate_output(1, {batch_size, map_h, map_w, n_links_}, &link_status));
    // output 2, [batch, map_h, map_w, 5], local gt_rboxes
    Tensor* gt_offsets = nullptr;
    OP_REQUIRES_OK(context,
      context->allocate_output(2, {batch_size, map_h, map_w, offsets_dim_}, &gt_offsets));
    // output 3, [batch, map_h, map_w] match groundtruth indices
    Tensor* match_indices = nullptr;
    OP_REQUIRES_OK(context,
      context->allocate_output(3, {batch_size, map_h, map_w}, &match_indices));

    // compute
    EncodeGroundtruthBatch(gt_rboxes.tensor<T, 3>(),
                           gt_counts.tensor<int, 1>(),
                           map_size.tensor<int, 1>(),
                           image_size.tensor<int, 1>(),
                           match_status_below.tensor<int, 3>(),
                           match_indices_below.tensor<int, 3>(),
                           match_status->tensor<int, 3>(),
                           link_status->tensor<int, 4>(),
                           gt_offsets->tensor<T, 4>(),
                           match_indices->tensor<int, 3>());
  }

private:
  /**
   * @brief Encode groundtruth rboxes to local groundtruths in a batch
   * @param gt_rboxes, tensor [batch, n_gt_max, rbox_dim]
   * @param gt_counts, int tensor [batch]
   * @param map_size, int tensor, size of feature maps [3]
   * @param image_size, int tensor, image size [3]
   * @param match_status_below, int tensor [batch, below_h, below_w] or [1,1,1]
   * @param match_indices_below, int tensor [batch, below_h, below_w] or [1,1,1]
   * @param match_status, int tensor [batch, map_h, map_w]
   * @param link_status, int tensor [batch, map_h, map_w, n_links_]
   * @param gt_offsets, tensor [batch, map_h, map_w, rbox_dim_]
   * @param match_indices, int tensor [batch, map_h, map_w]
   */
  void EncodeGroundtruthBatch(typename TTypes<T, 3>::ConstTensor gt_rboxes,
                              typename TTypes<int, 1>::ConstTensor gt_counts,
                              typename TTypes<int, 1>::ConstTensor map_size,
                              typename TTypes<int, 1>::ConstTensor image_size,
                              typename TTypes<int, 3>::ConstTensor match_status_below,
                              typename TTypes<int, 3>::ConstTensor match_indices_below,
                              typename TTypes<int, 3>::Tensor match_status,
                              typename TTypes<int, 4>::Tensor link_status,
                              typename TTypes<T, 4>::Tensor gt_offsets,
                              typename TTypes<int, 3>::Tensor match_indices) {
    const int batch_size = gt_rboxes.dimension(0);
    const int n_gt_max = gt_rboxes.dimension(1);
    const int map_h = map_size(0);
    const int map_w = map_size(1);
    const int below_h = match_status_below.dimension(1);
    const int below_w = match_status_below.dimension(2);
    const int image_h = image_size(0);
    const int image_w = image_size(1);

    assert(match_indices_below.dimension(1) == below_h);
    assert(match_indices_below.dimension(2) == below_w);

    for (int i = 0; i < batch_size; ++i) {
      const T* gt_i_data = gt_rboxes.data() + i * n_gt_max * rbox_dim_;
      const int gt_count = gt_counts(i);
      const int* match_status_below_i_data = cross_links_ ?
        match_status_below.data() + i * below_h * below_w : nullptr;
      const int* match_indices_below_i_data = cross_links_ ?
        match_indices_below.data() + i * below_h * below_w : nullptr;
      int* match_status_i_data = match_status.data() + i * map_h * map_w;
      int* link_status_i_data = link_status.data() + i * map_h * map_w * n_links_;
      T* gt_offsets_i_data = gt_offsets.data() + i * map_h * map_w * offsets_dim_;
      int* match_indices_i_data = match_indices.data() + i * map_h * map_w;

      EncodeGroundtruthExample(gt_i_data, gt_count,
                               match_status_below_i_data,
                               match_indices_below_i_data,
                               map_h, map_w,
                               below_h, below_w,
                               image_h, image_w,
                               match_status_i_data,
                               link_status_i_data,
                               gt_offsets_i_data,
                               match_indices_i_data);
    }
  }

  /**
   * @brief Encode groundtruth rboxes to local gt
   * @param gt_rboxes, tensor data [n_gt_max, gt_dim]
   * @param gt_count, int, number of groundtruths
   * @param match_status_below, int tensor data [map_h, map_w]
   * @param match_indicies_below, int tensor data [map_h, map_w]
   * @param gt_dim, int, groundtruth dimension
   * @param map_h, map_w, int, map width and height
   * @param below_h, below_w, int, below map width and height
   * @param image_h, image_w, int, image height
   * @param match_status, int tensor data [map_h, map_w]
   * @param link_status, int tensor data [map_h, map_w, n_links_]
   * @param gt_offsets, tensor data [map_h, map_w, rbox_dim_]
   * @param match_indices, int tensor data [map_h, map_w]
   */
  void EncodeGroundtruthExample(const T* gt_rboxes, const int gt_count,
                                const int* match_status_below,
                                const int* match_indicies_below,
                                const int map_h, const int map_w,
                                const int below_h, const int below_w,
                                const int image_h, const int image_w,
                                int* match_status,
                                int* link_status,
                                T* gt_offsets,
                                int* match_indices) {
    const T step_x = static_cast<T>(image_w) / map_w;
    const T step_y = static_cast<T>(image_h) / map_h;

    // compute node status
    for (int p = 0; p < map_h * map_w; ++p) {
      // find matching groundtruth
      const T anchor_cx = step_x * (static_cast<T>(p % map_w) + 0.5);
      const T anchor_cy = step_y * (static_cast<T>(p / map_w) + 0.5);

      int match, match_gt_idx;
      MatchGtRboxes(gt_rboxes, gt_count, anchor_cx, anchor_cy, &match, &match_gt_idx);

      match_status[p] = match;
      match_indices[p] = match_gt_idx;

      // project groundtruth to offsets
      T* gt_offsets_p = gt_offsets + p * offsets_dim_;
      if (match == 1) {
        const T* match_gt_rbox = gt_rboxes + match_gt_idx * rbox_dim_;
        CalculateOffsets(match_gt_rbox, anchor_cx, anchor_cy, gt_offsets_p);
      } else {
        for (int i = 0; i < offsets_dim_; ++i) {
          gt_offsets_p[i] = 0;
        }
      }
    }

    // compute link status
    for (int p = 0; p < map_h * map_w; ++p) {
      int* link_status_p = link_status + p * n_links_;
      int px = p % map_w;
      int py = p / map_w;

      // compute local links
      int link_idx = 0;
      for (int ny = py - 1; ny <= py + 1; ++ny) {
        for (int nx = px - 1; nx <= px + 1; ++nx) {
          if (nx == px && ny == py) {
            // skip self link
            continue;
          }
          int link = 0;
          int np = ny * map_w + nx;
          if (nx < 0 || nx >= map_w || ny < 0 || ny >= map_h) {
            // negative if link is out of the boundary
            link = -1;
          } else if (match_indices[p] >= 0 && match_indices[np] >= 0 &&
                     match_indices[p] != match_indices[np]) {
            // negative if two nodes have different matches
            link = -1;
          } else if (match_status[p] == -1 || match_status[np] == -1) {
            // negative if either the connecting anchor is negative,
            link = -1;
          } else if (match_status[p] == 1 && match_status[np] == 1 &&
                     match_indices[p] == match_indices[np]) {
            // link is positive if its two connecting nodes are both positive
            // and are matched to the same groundtruth
            link = 1;
          } else {
            // otherwise, link is marked as 'ignored'
            link = 0;
          }
          link_status_p[link_idx] = link;
          link_idx++;
        }
      }
      assert(link_idx == n_within_links_);

      // compute cross links
      if (cross_links_) {
        int y_start = std::min(cross_stride_ * py, below_h - cross_stride_);
        int y_end = std::min(cross_stride_ * (py + 1), below_h);
        int x_start = std::min(cross_stride_ * px, below_w - cross_stride_);
        int x_end = std::min(cross_stride_ * (px + 1), below_w);
        int link_idx = 0;
        for (int below_y = y_start; below_y < y_end; ++below_y) {
          for (int below_x = x_start; below_x < x_end; ++below_x) {
            int bp = below_y * below_w + below_x;
            int link = 0;
            if (match_status[p] == -1 || match_status_below[bp] == -1) {
              // negative if either node is negative
              link = -1;
            } else if (false) { // FIXME
              // negative if to nodes have different matches
              link = -1;
            } else if (match_status[p] == 1 && match_status_below[bp] == 1 &&
                       match_indices[p] == match_indicies_below[bp]) {
              // positive if both nodes are positive and are matched to the same gt
              link = 1;
            } else {
              // ignored otherwise
              link = 0;
            }
            link_status_p[n_within_links_ + link_idx] = link;
            link_idx++;
          }
        }
        assert(link_idx == n_cross_links_);
      }
    }
  }

  /**
   * @brief Encode groundtruth into regression targets of a local region
   * @param gt_rboxes, tensor data [n_gt_max, rbox_dim_]
   * @param gt_count, int, number of groundtruths
   * @param anchor_cx, anchor center x
   * @param anchor_cy, anchor center y
   * @param match_data, match status data [1]
   * @param match_gt_idx_data, match gt index data [1]
   */
  void MatchGtRboxes(const T* gt_rboxes, int gt_count,
                     T anchor_cx, T anchor_cy,
                     int* match_data, int* match_gt_idx_data) {
    T min_scale_diff = std::numeric_limits<T>::infinity();
    int match_gt_idx = -1;

    for (int i = 0; i < gt_count; ++i) {
      const T* gt_rboxes_i = gt_rboxes + i * rbox_dim_;
      T gt_height = gt_rboxes_i[3];
      T scale_diff = std::max(anchor_size_ / gt_height, gt_height / anchor_size_);
      if (scale_diff < neg_scale_diff_thresh_) {
        T dist_x, dist_y;
        bool is_inside = util::point_inside_rbox(gt_rboxes_i, anchor_cx, anchor_cy,
                                                 &dist_x, &dist_y); // TODO: add a margin
        if (is_inside) {
          if (scale_diff < min_scale_diff) {
            match_gt_idx = i;
            min_scale_diff = scale_diff;
          }
        }
      }
    }

    int match;
    if (match_gt_idx == -1) {
      match = -1;
    } else if (min_scale_diff < pos_scale_diff_thresh_) {
      match = 1;
    } else {
      match = 0;
    }

    *match_data = match;
    *match_gt_idx_data = match_gt_idx;
  }

  void CalculateOffsets(const T* gt_rbox, T anchor_cx, T anchor_cy, T* gt_offsets_p) {
    const T eps = 1e-6;
    const T half = static_cast<T>(0.5);
    T gt_cx     = gt_rbox[0];
    T gt_cy     = gt_rbox[1];
    T gt_width  = gt_rbox[2];
    T gt_height = gt_rbox[3];
    T gt_theta  = gt_rbox[4];

    // region bounding box
    T anchor_box[4] = {anchor_cx - half * anchor_size_, anchor_cy - half * anchor_size_,
                       anchor_cx + half * anchor_size_, anchor_cy + half * anchor_size_};

    // rotate groundtruth center around the local center anti-clockwisely by gt_theta
    T hgt_cx, hgt_cy;
    util::rotate_around(gt_cx, gt_cy, anchor_cx, anchor_cy, -gt_theta, &hgt_cx, &hgt_cy);

    // groundtruth box rotated to the horizontal direction
    T hgt_bbox[4] = {hgt_cx - half * gt_width, hgt_cy - half * gt_height,
                     hgt_cx + half * gt_width, hgt_cy + half * gt_height};

    // clip the width of the rotated groundtruth
    T clipper_bbox[4] = {anchor_box[0], -std::numeric_limits<T>::infinity(),
                         anchor_box[2],  std::numeric_limits<T>::infinity()};
    T clipped_bbox[4];
    util::bbox_intersection(clipper_bbox, hgt_bbox, clipped_bbox);

    // rotate the clipped box back (clockwisely)
    T clipped_cx = half * (clipped_bbox[0] + clipped_bbox[2]);
    T clipped_cy = half * (clipped_bbox[1] + clipped_bbox[3]);
    T segment_cx, segment_cy;
    util::rotate_around(clipped_cx, clipped_cy, anchor_cx, anchor_cy, gt_theta, &segment_cx, &segment_cy);
    T segment_width = clipped_bbox[2] - clipped_bbox[0];
    T segment_height = clipped_bbox[3] - clipped_bbox[1];

    gt_offsets_p[0] = (segment_cx - anchor_cx) / anchor_size_;
    gt_offsets_p[1] = (segment_cy - anchor_cy) / anchor_size_;
    gt_offsets_p[2] = std::log(eps + segment_width / anchor_size_);
    gt_offsets_p[3] = std::log(eps + segment_height / anchor_size_);
    gt_offsets_p[4] = std::cos(gt_theta);
    gt_offsets_p[5] = std::sin(gt_theta);
  }

  const int rbox_dim_;
  const int offsets_dim_;
  const int n_within_links_;
  const int n_cross_links_;
  const int cross_stride_;

  bool cross_links_;
  int n_links_;
  T anchor_size_;
  T pos_scale_diff_thresh_;
  T neg_scale_diff_thresh_;
};

REGISTER_KERNEL_BUILDER(Name("EncodeGroundtruth").Device(DEVICE_CPU),
                        EncodeGroundtruthOp<CPUDevice, float>)
