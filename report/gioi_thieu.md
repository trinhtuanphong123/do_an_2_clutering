% ============================================================
%  Chương 1 — Giới thiệu
%  File: chuong_1.tex
% ============================================================

%\chapter{Giới thiệu}
%\label{chap:intro}

% ────────────────────────────────────────────────────────────
\section*{Động cơ nghiên cứu}
\label{sec:motivation}

Trong bối cảnh nền kinh tế số phát triển nhanh chóng, ngành tài chính
ngân hàng đang đối mặt với lượng dữ liệu giao dịch khổng lồ được sinh ra
mỗi ngày từ hàng triệu chủ thẻ tín dụng.
Việc hiểu rõ hành vi tiêu dùng, thói quen thanh toán và mức độ rủi ro
tín dụng của từng nhóm khách hàng không còn là lợi thế cạnh tranh mà đã
trở thành yêu cầu thiết yếu đối với các tổ chức tài chính hiện đại
\cite{hand2001principles}.
Tuy nhiên, phần lớn dữ liệu khách hàng trong thực tế tồn tại dưới dạng không có nhãn: không có thông tin định sẵn về việc
khách hàng thuộc nhóm nào hay hành vi của họ mang đặc điểm gì.
Điều này khiến các phương pháp học có giám sát truyền thống trở nên
không áp dụng được trực tiếp.

Học máy không giám sát, cụ thể là các phương pháp phân cụm (clustering),
cung cấp một hướng tiếp cận phù hợp để khám phá cấu trúc ẩn trong dữ
liệu mà không cần nhãn định sẵn \cite{jain2010data}.
Phân cụm khách hàng dựa trên hành vi sử dụng thẻ tín dụng cho phép các
ngân hàng và tổ chức tín dụng nhóm các khách hàng có đặc điểm tương đồng
lại với nhau, từ đó xây dựng các chiến lược marketing có mục tiêu, thiết
kế sản phẩm tài chính phù hợp với từng phân khúc, và quản lý rủi ro hiệu
quả hơn \cite{ngai2009application}.

Trong số các thuật toán phân cụm, K-Means là lựa chọn phổ biến nhất nhờ
tính đơn giản và hiệu quả tính toán .
Tuy nhiên, K-Means đưa ra một số giả thiết hạn chế về cấu trúc dữ liệu,
đặc biệt là giả thiết về hình dạng cụm hình cầu và kích thước đồng nhất.
Mô hình hỗn hợp Gaussian (Gaussian Mixture Model, GMM) là một hướng tiếp
cận xác suất linh hoạt hơn, cho phép mô hình hóa các cụm có hình dạng
ellipse và cung cấp độ không chắc chắn về nhãn cụm cho từng điểm dữ liệu
\cite{bishop2006pattern}.

Xuất phát từ những nhận định trên, đồ án này được thực hiện nhằm nghiên
cứu và so sánh hai phương pháp phân cụm K-Means và GMM trong bài toán
phân khúc khách hàng thẻ tín dụng, đồng thời xây dựng một quy trình
phân tích dữ liệu hoàn chỉnh từ kiểm tra và làm sạch dữ liệu, kỹ thuật
đặc trưng, huấn luyện và đánh giá mô hình, đến diễn giải kết quả phân
cụm theo ngữ cảnh nghiệp vụ.

% ────────────────────────────────────────────────────────────
\section*{Các nghiên cứu liên quan}
\label{sec:related_work}

Bài toán phân khúc khách hàng tài chính dựa trên dữ liệu hành vi đã thu
hút sự quan tâm đáng kể trong cộng đồng nghiên cứu học máy và khai phá
dữ liệu trong những năm gần đây.

Ngain và các cộng sự \cite{ngai2009application} thực hiện
một tổng quan hệ thống về ứng dụng khai phá dữ liệu trong lĩnh vực
marketing tài chính, chỉ ra rằng phân cụm là kỹ thuật được sử dụng phổ
biến nhất trong các bài toán phân khúc khách hàng, chiếm tỷ lệ cao trong
tổng số các công trình được khảo sát.
Nghiên cứu nhấn mạnh tầm quan trọng của việc kết hợp kiến thức nghiệp vụ
vào quá trình diễn giải kết quả phân cụm.

Trong lĩnh vực tín dụng cụ thể,Hand và các cộng sự \cite{hand2001principles}
 đặt nền tảng lý thuyết cho việc sử dụng các
phương pháp thống kê và học máy trong phân tích dữ liệu thẻ tín dụng,
bao gồm phân tích hành vi thanh toán, đánh giá rủi ro và phát hiện gian
lận.

Về phía phương pháp luận, Jain và các cộng sự \cite{jain2010data}
cung cấp một khảo sát toàn diện về các thuật toán phân cụm trong khai
phá dữ liệu, trong đó K-Means được đánh giá là thuật toán có ảnh hưởng
nhất trong 50 năm qua nhờ tính đơn giản và khả năng mở rộng.
Đối với GMM, Fraley và Raftery  \cite{fraley2002model} đề xuất
khuôn khổ phân cụm dựa trên mô hình (model-based clustering) sử dụng
hỗn hợp Gaussian với tiêu chí BIC để lựa chọn số thành phần, được áp
dụng rộng rãi trong nhiều lĩnh vực ứng dụng.

Về vấn đề đánh giá chất lượng phân cụm,Arbelaitz và các cộng sự \cite{arbelaitz2013extensive}
 thực hiện đánh giá so sánh toàn diện trên
 chỉ số nội tại và kết luận rằng không có
chỉ số đơn lẻ nào vượt trội hoàn toàn — việc kết hợp nhiều chỉ số như
Silhouette, Davies-Bouldin và Calinski-Harabasz cho kết quả đánh giá
đáng tin cậy hơn, đây cũng là chiến lược được áp dụng trong đồ án này.

Điểm khác biệt của đồ án so với các nghiên cứu đã có là sự kết hợp có
hệ thống giữa: (i) quy trình kiểm tra và làm sạch dữ liệu nghiêm ngặt
với các ràng buộc miền nghiệp vụ, (ii) kỹ thuật đặc trưng có chủ đích
dựa trên hiểu biết về dữ liệu tài chính, (iii) so sánh đồng thời K-Means
và GMM với chiến lược lựa chọn mô hình đa chỉ số, và (iv) đặt tên phân
khúc khách hàng dựa trên quy tắc nghiệp vụ có thể diễn giải được.

% ────────────────────────────────────────────────────────────
\section*{Đối tượng và phạm vi nghiên cứu}
\label{sec:scope}

\paragraph{Đối tượng nghiên cứu.}
Đồ án tập trung nghiên cứu hai nhóm đối tượng chính:
\begin{enumerate}
    \item \textbf{Phương pháp}: Hai thuật toán phân cụm K-Means và Mô
    hình hỗn hợp Gaussian (GMM), bao gồm cơ sở lý thuyết, quy trình
    huấn luyện, các tiêu chí lựa chọn mô hình và phương pháp đánh giá
    chất lượng phân cụm nội tại.

    \item \textbf{Dữ liệu}: Bộ dữ liệu \emph{Credit Card Dataset for
    Clustering} \cite{ccdata_kaggle}, bao gồm thông tin hành vi sử dụng
    thẻ tín dụng của khoảng 9.000 chủ thẻ tích cực trong vòng 6 tháng,
    với 18 biến hành vi bao gồm số dư tài khoản, lượng giao dịch mua
    sắm, tần suất sử dụng, ứng tiền mặt, hạn mức tín dụng và lịch sử
    thanh toán.
\end{enumerate}

\paragraph{Phạm vi nghiên cứu.}
Đồ án giới hạn trong phạm vi sau:
\begin{itemize}
    \item Chỉ xem xét các phương pháp phân cụm không giám sát; không
    đề cập đến các bài toán phân loại hay hồi quy có giám sát.

    \item Tập trung vào hai thuật toán K-Means và GMM; các thuật toán
    phân cụm khác (DBSCAN, phân cụm phân cấp, v.v.) không nằm trong
    phạm vi triển khai thực nghiệm mặc dù có thể được đề cập ở mức
    so sánh lý thuyết.

    \item Chỉ sử dụng các chỉ số đánh giá nội tại  do không có nhãn thực trong bộ dữ liệu.

    \item Bài toán đặt trong ngữ cảnh phân khúc khách hàng phục vụ mục
    đích phân tích hành vi; không đề cập đến các bài toán dự đoán rủi
    ro tín dụng hay phát hiện gian lận.
\end{itemize}

% ────────────────────────────────────────────────────────────
\section*{Mục tiêu nghiên cứu}
\label{sec:objectives}

Đồ án hướng đến ba mục tiêu cụ thể như sau:

\begin{enumerate}
    \item \textbf{Nghiên cứu lý thuyết}: Hệ thống hóa cơ sở lý thuyết
    của hai phương pháp phân cụm K-Means và GMM, bao gồm bài toán tối
    ưu hóa, thuật toán huấn luyện, giả thiết mô hình và các tiêu chí
    lựa chọn số cụm; đồng thời tổng hợp các chỉ số đánh giá chất lượng
    phân cụm nội tại được sử dụng rộng rãi trong cộng đồng nghiên cứu.

    \item \textbf{Xây dựng quy trình thực nghiệm}: Thiết kế và triển
    khai một quy trình phân tích dữ liệu hoàn chỉnh trên bộ dữ liệu
    thẻ tín dụng, bao gồm kiểm tra chất lượng dữ liệu, xử lý dữ liệu
    khuyết, kỹ thuật đặc trưng có định hướng nghiệp vụ, loại bỏ đặc
    trưng tương quan, chuẩn hóa và biến đổi phân phối, huấn luyện mô
    hình với nhiều cấu hình, và đánh giá độ ổn định.

    \item \textbf{Phân tích và diễn giải kết quả}: So sánh hiệu năng
    của K-Means và GMM thông qua bộ chỉ số đa chiều (Silhouette Score,
    Davies-Bouldin Index, Calinski-Harabasz Index), đặt tên và mô tả
    đặc điểm của từng phân khúc khách hàng theo ngữ cảnh nghiệp vụ tài
    chính, từ đó rút ra các nhận xét và đề xuất ứng dụng thực tiễn.
\end{enumerate}

% ────────────────────────────────────────────────────────────
\section*{Kết cấu đồ án}
\label{sec:structure}

Phần còn lại của đồ án được tổ chức như sau:

\textbf{Chương~1 — Cơ sở lý thuyết} trình bày nền tảng kiến thức cần
thiết cho đồ án, bao gồm tổng quan về học máy không giám sát và bài toán
phân cụm; các độ đo khoảng cách; lý thuyết về thuật toán K-Means và Mô
hình hỗn hợp Gaussian; và các chỉ số đánh giá chất lượng phân cụm nội
tại.

\textbf{Chương~2 — Thực nghiệm và đánh giá} 

