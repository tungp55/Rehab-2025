# Phân tích độ phức tạp tính toán – merge_tracks_org.py

## Ký hiệu

- **N**: số dòng (detection) trong file input.
- **T**: số track (đỉnh đồ thị), tức `max_track_id`.
- **F**: số frame (ví dụ `nframes` khi ghi output).

## Độ phức tạp từng bước

| Bước | Thời gian | Ghi chú |
|------|-----------|--------|
| `read_frame_data` | O(N) | Đọc file, tách dòng. |
| `convert_2_track_data` | O(N) | Một lần duyệt; mở rộng mảng track O(T) amortized. |
| `calculate_maximum_values` | O(T²) | Hai vòng lặp lồng nhau trên cặp track; mỗi lần gọi `centroid_distance` O(1). |
| `build_graph` | O(T²) | Thêm T đỉnh; O(T²) cặp (u,v), mỗi cặp kiểm tra overlap và cost O(1). |
| Bellman–Ford (NetworkX) | O(T³) | Đồ thị V = T+1, E = O(T²); thuật toán O(V·E) = O(T³). |
| `write_tracks_chain_to_file` | O(F + N) | Duyệt theo frame và ghi theo detection; với index hợp lý là O(F + N). |

## Tổng kết

- **Thời gian:** O(N + T² + T³) (bỏ qua bước ghi file). Khi T lớn, **bước tìm đường đi (Bellman–Ford) chi phối: O(T³)**.
- **Không gian:** O(N) cho `frames_data`, O(T) cho `tracks_data`, O(T²) cho đồ thị và O(T) cho đường đi → **O(N + T²)**.

---

## Đoạn văn cho bài báo (tiếng Anh)

The merge-tracks stage first builds a directed acyclic graph over track segments, with edge weights reflecting temporal non-overlap and transition cost; then it finds a minimum-cost path from a fixed source track to a sink using the Bellman–Ford algorithm. The graph has *V* = *T* + 1 vertices and *O*(*T*²) edges, where *T* is the number of track segments, so the overall run time is *O*(*N* + *T*²) for reading and graph construction and *O*(*T*³) for shortest-path computation. Space complexity is *O*(*N* + *T*²) for the input detections and the graph. Thus the proposed post-processing is cubic in the number of track segments and linear in the number of detections, and remains tractable for typical clinical sequences where *T* is moderate.

---

## Đoạn văn cho bài báo (tiếng Việt)

Giai đoạn gộp track xây dựng một đồ thị có hướng không chu trình trên các đoạn track, với trọng số cạnh phản ánh sự không chồng lấn thời gian và chi phí chuyển tiếp, sau đó tìm đường đi có tổng chi phí nhỏ nhất từ một track nguồn cố định tới đỉnh sink bằng thuật toán Bellman–Ford. Đồ thị có *V* = *T* + 1 đỉnh và *O*(*T*²) cạnh, với *T* là số đoạn track, nên thời gian chạy tổng cộng là *O*(*N* + *T*²) cho bước đọc dữ liệu và xây đồ thị và *O*(*T*³) cho bước tìm đường đi ngắn nhất. Độ phức tạp bộ nhớ là *O*(*N* + *T*²) cho dữ liệu detection và đồ thị. Như vậy, hậu xử lý gộp track có độ phức tạp bậc ba theo số đoạn track và tuyến tính theo số detection, vẫn khả thi với các chuỗi lâm sàng thông thường khi *T* có độ lớn vừa phải.
