#pragma once

#include "mecan/tensor.h"
#include <cstddef>
#include <vector>

namespace mecan {
namespace vision {
namespace vector {

    struct Point2D {
        float x, y;
    };

    struct LineSegment {
        Point2D start, end;
    };

    /**
     * Extract Vertices (Corners / Junctions)
     * Takes a binary edge map and returns the (x, y) coordinates of junction points.
     * Output must be pre-allocated. Returns the number of vertices found.
     */
    size_t extract_vertices(
        const float* edge_map, size_t H, size_t W,
        Point2D* out_vertices, size_t max_vertices
    );

    /**
     * Fit Lines to Edges (Hough/RANSAC Hybrid Approximation)
     * Converts raw pixel chains into mathematical LineSegments.
     */
    size_t fit_lines(
        const float* edge_map, size_t H, size_t W,
        LineSegment* out_segments, size_t max_segments,
        float tolerance = 2.0f
    );

    /**
     * Douglas-Peucker Polygonization
     * Simplifies a continuous edge contour into a polygon with fewer vertices.
     * pts: input ordered contour points
     * out_pts: output simplified polygon points
     */
    size_t polygonize(
        const Point2D* pts, size_t num_pts,
        Point2D* out_pts, size_t max_out,
        float epsilon
    );

    /**
     * Signed Distance Field (SDF)
     * Computes the distance from every pixel to the nearest edge/boundary.
     * Essential for vector graphics rendering and font processing.
     */
    void signed_distance_field(
        const float* binary_mask, float* out_sdf,
        size_t H, size_t W
    );

} // namespace vector
} // namespace vision
} // namespace mecan
