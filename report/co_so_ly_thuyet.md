\section{Học máy không giám sát}
\label{sec:unsupervised_learning}

Học máy không giám sát (Unsupervised Learning) là một trong những phân nhánh nền tảng của trí tuệ nhân tạo và học máy toàn cục \cite{hastie2009elements, bishop2006pattern}. Khác biệt cốt lõi của học không giám sát so với học có giám sát (Supervised Learning) nằm ở bản chất của tập dữ liệu huấn luyện. Trong ngữ cảnh này, hệ thống không được cung cấp các cặp dữ liệu vào-ra định sẵn $(\mathbf{x}_i, y_i)$, trong đó $y_i$ là nhãn mục tiêu hoặc giá trị đích cần dự báo. Thay vào đó, tập dữ liệu huấn luyện $\mathcal{D}$ chỉ bao gồm một tập hợp các vectơ quan sát biểu diễn các thực thể:
\begin{equation}
\mathcal{D} = \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N\}
\end{equation}
Trong đó, mỗi quan sát $\mathbf{x}_i \in \mathbb{R}^D$ là một vectơ đặc trưng $D$ chiều nằm trong không gian thực thể \cite{murphy2012machine}. 

Mục tiêu của học không giám sát không phải là thiết lập một hàm ánh xạ tuyến tính hoặc phi tuyến để tối thiểu hóa sai số dự báo, mà là tự động khám phá các cấu trúc ẩn, các mối quan hệ nội tại, hoặc quy luật phân phối xác suất thống kê tiềm ẩn bên trong cấu trúc dữ liệu bản sinh \cite{murphy2012machine, ghahramani2003unsupervised}. Về mặt toán học thống kê, bài toán này thường có thể được mô hình hóa dưới dạng ước lượng mật độ xác suất bản sinh của dữ liệu, ký hiệu là $p(\mathbf{x})$ \cite{bishop2006pattern}. 

Do không có các giám sát có thể định hướng bằng nhãn chuẩn, học không giám sát đối mặt với những thách thức lớn về mặt lý thuyết lẫn thực tiễn \cite{ghahramani2003unsupervised}:
\begin{itemize}
    \item Tính bất định trong tiêu chí tối ưu: Không tồn tại một hàm mất mát trực quan duy nhất (như sai số bình phương trung bình hay entropy chéo) để đánh giá tuyệt đối hiệu năng của mô hình. Các độ đo tối ưu thường mang tính chất hình học nội bộ hoặc dựa trên các tiêu chí thông tin thống kê.
    \item Thách thức về mặt diễn giải: Kết quả đầu ra của các thuật toán không giám sát đòi hỏi người nghiên cứu hoặc các chuyên gia phân tích nghiệp vụ phải thực hiện ánh xạ ngược lại không gian ngữ nghĩa thực tế để gán ý nghĩa cho cấu trúc tìm được.
\end{itemize}

Trong thực tiễn nghiên cứu khoa học dữ liệu, các phương pháp học không giám sát chủ yếu được phân tách thành các bài toán lớn sau \cite{hastie2009elements, murphy2012machine}:
\begin{enumerate}
    \item Phân cụm dữ liệu (Clustering): Phân rã tập dữ liệu thành các tập hợp con sao cho các thực thể trong cùng một nhóm có độ tương đồng cao, và giữa các nhóm có độ tách biệt rõ rệt theo một hệ số khoảng cách xác định.
    \item Giảm chiều dữ liệu (Dimensionality Reduction): Chiếu dữ liệu từ không gian cao chiều $D$ xuống một không gian biến ẩn thấp chiều $K$ ($K \ll D$) mà vẫn bảo toàn tối đa lượng thông tin hoặc phương sai của dữ liệu gốc.
    \item Ước lượng phân phối mật độ (Density Estimation): Xác định hình dạng toán học của hàm mật độ xác suất để kiểm tra cấu trúc hình học của không gian mẫu.
    \item Khai phá luật kết hợp (Association Rule Mining): Tìm kiếm các mối quan hệ phụ thuộc đồng thời giữa các thuộc tính trong cơ sở dữ liệu lớn.
\end{enumerate}

Trong giới hạn của nghiên cứu này, chúng tôi tập trung đào sâu vào bài toán phân cụm dữ liệu — một nhánh tiếp cận mang tính chất trực quan và thực tiễn bậc nhất để bóc tách cấu trúc phân khúc của thực thể.


\section{Bài toán phân cụm dữ liệu}
\label{sec:clustering_problem}

\subsection{Định nghĩa và phân loại}
\label{subsec:definition_classification}

Phân cụm dữ liệu là một trong những kỹ thuật cốt lõi và được áp dụng rộng rãi nhất của học không giám sát. Về mặt phân tích toán học, cho trước một tập dữ liệu $\mathcal{D} = \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N\}$ gồm $N$ quan sát trong không gian $D$ chiều, bài toán phân cụm nhằm mục đích phân rã $\mathcal{D}$ thành một họ gồm $K$ tập hợp con (gọi là các cụm) $\mathcal{C} = \{C_1, C_2, \dots, C_K\}$ \cite{jain1999data}. Quá trình phân rã này tuân theo hai nguyên lý hình học không gian nền tảng:
\begin{itemize}
    \item Độ tương đồng nội cụm (Homogeneity): Các phần tử thuộc cùng một cụm $C_k$ phải có độ tương đồng tối đa theo một cấu trúc khoảng cách quy định.
    \item Độ tách biệt ngoại cụm (Separation): Các phần tử thuộc các cụm khác nhau ($C_i$ và $C_j$ với $i \neq j$) phải có độ biệt lập hoặc khoảng cách lớn nhất có thể.
\end{itemize}

Dựa trên cấu trúc topo của không gian phân hoạch và cơ chế gán nhãn của thuật toán, các phương pháp phân cụm trong học thuật hiện đại được chia thành hai trường phái tiếp cận chính \cite{jain1999data}:

\begin{enumerate}
    \item Phân cụm phân hoạch cứng : Mỗi quan sát $\mathbf{x}_i$ chỉ được phép thuộc về duy nhất một cụm xác định. Nói cách khác, các cụm tạo thành một phân hoạch không gian độc lập, thỏa mãn điều kiện:
    \begin{equation}
    C_i \cap C_j = \emptyset \quad \forall i \neq j \quad \text{và} \quad \bigcup_{k=1}^K C_k = \mathcal{D}
    \end{equation}
    Đại diện kinh điển cho trường phái này là thuật toán K-means \cite{macqueen1967some}, hoạt động dựa trên việc tối thiểu hóa tổng bình phương khoảng cách đến các tâm cụm đại diện.

    \item Phân cụm mềm hoặc phân cụm mờ : Mỗi quan sát $\mathbf{x}_i$ không bị ràng buộc cố định mà sở hữu một phân phối xác suất hoặc mức độ thành viên thuộc về tất cả các cụm. Hệ số thành viên $w_{ik}$ biểu thị xác suất điểm $\mathbf{x}_i$ nằm trong cụm $C_k$, thỏa mãn ràng buộc:
    \begin{equation}
    \sum_{k=1}^K w_{ik} = 1 \quad \text{và} \quad 0 \leq w_{ik} \leq 1
    \end{equation}
    Mô hình hỗn hợp Gauss (Gaussian Mixture Models) phối hợp với thuật toán tối ưu hóa kỳ vọng là đại diện tiêu biểu cho hướng tiếp cận dựa trên lý thuyết phân phối xác suất thống kê này.
\end{enumerate}


\subsection{Tiêu chí chất lượng phân cụm}
\label{subsec:clustering_quality_criteria}

Để xác định một cấu trúc phân cụm thế nào là tối ưu về mặt lý thuyết thống kê và hình học không gian, người ta không dựa trên các nhãn mục tiêu ngoại vi mà thiết lập các tiêu chí toán học trừu tượng dựa trên cấu trúc khoảng cách nội tại của tập dữ liệu. Về mặt lý thuyết, một phân hoạch không gian được coi là tốt khi và chỉ khi nó thỏa mãn đồng thời hai tính chất hình học đối nghịch nhưng bổ trợ sau đây \cite{hastie2009elements}:

\begin{enumerate}
    \item Tính gắn kết nội cụm : Tiêu chí này yêu cầu các quan sát được xếp vào cùng một cụm $C_k$ phải nằm gần nhau nhất có thể trong không gian đặc trưng. Về mặt toán học, độ phân tán nội cụm biểu thị sự cô đặc của các phần tử xung quanh một đại diện cụm (thường là tâm cụm hoặc các điểm lân cận cốt lõi). Việc tối đa hóa tính gắn kết nội cụm đồng nghĩa với việc tối thiểu hóa hàm phương sai hình học nội bộ của cấu trúc đó.
    
    \item Tính tách biệt liên cụm : Tiêu chí này đòi hỏi các cụm độc lập ($C_i$ và $C_j$ với $i \neq j$) phải được đẩy ra xa nhau tối đa trong không gian đặc trưng. Khoảng cách biên giữa các cụm phải đủ lớn để đảm bảo không xảy ra hiện tượng chồng lấn không gian hành vi, tạo ra các ranh giới phân tách rõ rệt giữa các miền mật độ dữ liệu khác nhau.
\end{enumerate}

Để định lượng các tính chất trừu tượng trên thành một bài toán tối ưu hóa toán học, hệ thống cần thiết lập một hàm mục tiêu tổng quát (Objective function). Trong lý thuyết phân cụm hình học, hàm mục tiêu kinh điển nhất được sử dụng để đánh giá chất lượng phân hoạch là hàm tổng bình phương khoảng cách nội cụm (Within-Cluster Sum of Squares - WCSS) \cite{tan2016introduction}.

Giả sử mỗi cụm $C_k$ được đại diện bởi một tâm hình học $\mathbf{m}_k$ (centroid), hàm mục tiêu tổng quát WCSS đo lường tổng độ lệch bình phương của tất cả các điểm dữ liệu đến tâm cụm tương ứng của chúng được mô tả thông qua công thức:
\begin{equation}
\Phi(\mathcal{C}) = \sum_{k=1}^K \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \mathbf{m}_k\|^2
\end{equation}
Trong đó, $\|\cdot\|$ ký hiệu cho dạng chuẩn Euclid toán học đại diện cho khoảng cách hình học. Một cấu trúc phân cụm lý tưởng về mặt lý thuyết sẽ là nghiệm của bài toán tối ưu hóa nhằm tìm ra phân hoạch $\mathcal{C}^*$ sao cho:
\begin{equation}
\mathcal{C}^* = \arg\min_{\mathcal{C}} \Phi(\mathcal{C})
\end{equation}

Hàm mục tiêu tổng quát này đóng vai trò là nền tảng khái niệm cốt lõi. Trong thực tế triển khai kỹ thuật, do không thể quan sát trực tiếp hàm mục tiêu trên toàn bộ các dạng topo dữ liệu phức tạp, các chỉ số đo lường thực nghiệm sẽ được phát triển ở các chương sau nhằm xấp xỉ và cụ thể hóa các tính chất gắn kết và tách biệt này thành các đại lượng số học có thể tính toán được.


\section{Độ đo khoảng cách}
\label{sec:distance_metrics}

Trong các thuật toán phân cụm hình học và mô hình xác suất bản sinh, việc định lượng độ tương đồng hoặc độ khác biệt giữa các quan sát là một bước đi mang tính quyết định. Các độ đo này thiết lập cơ sở toán học để xác định cấu trúc topo của không gian đặc trưng, trực tiếp tác động đến cấu trúc phân hoạch nội cụm và liên cụm.

\subsection{Khoảng cách Minkowski và các trường hợp đặc biệt}
\label{subsec:minkowski_distance}

Khoảng cách Minkowski là một dạng độ đo tổng quát hóa trong không gian định chuẩn $L_p$, đóng vai trò là cấu trúc nền tảng để suy dẫn ra các độ đo khoảng cách hình học thông dụng khác \cite{deza2016encyclopedia}. Cho trước hai vectơ đặc trưng $D$ chiều $\mathbf{x}_i = (x_{i1}, x_{i2}, \dots, x_{iD})^T$ và $\mathbf{x}_j = (x_{j1}, x_{j2}, \dots, x_{jD})^T$ thuộc không gian thực số $\mathbb{R}^D$, khoảng cách Minkowski bậc $p$ giữa hai điểm này được định nghĩa bằng biểu thức:
\begin{equation}
d_{\text{Minkowski}}(\mathbf{x}_i, \mathbf{x}_j) = \left( \sum_{d=1}^D |x_{id} - x_{jd}|^p \right)^{\frac{1}{p}}
\end{equation}
Trong đó, tham số $p \geq 1$ quy định cấu trúc hình học của không gian đo. Thay đổi giá trị của $p$ dẫn đến các trường hợp đặc biệt mang tính chất topo khác nhau:
\begin{itemize}
    \item Khi $p = 1$, khoảng cách Minkowski suy biến thành khoảng cách Manhattan. Độ đo này tính tổng độ lệch tuyệt đối giữa các tọa độ tương ứng trên từng chiều không gian độc lập, mô phỏng khoảng cách di chuyển theo cấu trúc lưới trực giao.
    \item Khi $p = 2$, độ đo này tương ứng với khoảng cách Euclidean kinh điển. Đây là dạng khoảng cách đo lường theo đường thẳng trực quan nhất trong không gian hình học phẳng.
\end{itemize}

\subsection{Khoảng cách Euclidean}
\label{subsec:euclidean_distance}

Là trường hợp đặc biệt khi bậc định chuẩn $p = 2$, khoảng cách Euclidean biểu thị độ dài ngắn nhất kết nối hai điểm dữ liệu trong không gian thực số hữu hạn \cite{deza2016encyclopedia}. Công thức xác định khoảng cách Euclidean giữa hai vectơ $\mathbf{x}_i$ và $\mathbf{x}_j$ được thiết lập dưới dạng chuẩn $L_2$:
\begin{equation}
d_{\text{Euclidean}}(\mathbf{x}_i, \mathbf{x}_j) = \|\mathbf{x}_i - \mathbf{x}_j\|_2 = \sqrt{\sum_{d=1}^D (x_{id} - x_{jd})^2}
\end{equation}
Về mặt tính chất toán học, khoảng cách Euclidean thỏa mãn đầy đủ bốn tiên đề chuẩn mực của một không gian metric bao gồm: tính không âm, tính đồng nhất phục vụ cho các điểm trùng nhau, tính đối xứng qua lại, và bất đẳng thức tam giác.

Độ đo này chiếm vị trí phổ biến nhất trong các thuật toán phân cụm như K-Means và mô hình xác suất Gaussian Mixture Models bởi mối liên hệ mật thiết giữa hình học Euclid và lý thuyết xác suất thống kê \cite{bishop2006pattern}:
\begin{itemize}
    \item Đối với thuật toán K-Means, việc tối thiểu hóa hàm mục tiêu tổng phương sai WCSS bản chất chính là việc tối thiểu hóa tổng bình phương khoảng cách Euclidean từ các điểm đến tâm cụm tương ứng. Cấu trúc này ngầm định các cụm dữ liệu có xu hướng phân bổ đều theo dạng hình cầu xung quanh tâm hình học.
    \item Đối với mô hình GMM, khoảng cách Euclidean xuất hiện trực tiếp trên số mũ của hàm mật độ xác suất Gauss chính tắc. Khi ma trận hiệp phương sai có dạng cấu trúc ma trận đơn vị nhân với một hệ số tỉ lệ hoặc dạng ma trận đường chéo, các đường đẳng mức mật độ xác suất sẽ có dạng hình cầu hoặc hình elip đồng trục. Lúc này, khoảng cách Euclidean đóng vai trò cốt lõi trong việc tính toán xác suất thành viên của các quan sát đối với từng thành phần hỗn hợp.
\end{itemize}

\subsection{Cosine Similarity}
\label{subsec:cosine_similarity}

Khác biệt với các tiếp cận dựa trên hệ số khoảng cách định chuẩn $L_p$ vốn đo lường độ lệch tuyệt đối về mặt vị trí không gian, độ tương đồng Cosine tập trung đo lường xu hướng hướng trục thông qua góc kẹp giữa hai vectơ đặc trưng trong không gian tích vô hướng \cite{tan2016introduction}. Hệ số tương đồng Cosine giữa $\mathbf{x}_i$ và $\mathbf{x}_j$ được xác định bởi công thức:
\begin{equation}
\text{Sim}_{\text{Cosine}}(\mathbf{x}_i, \mathbf{x}_j) = \cos(\theta) = \frac{\mathbf{x}_i^T \mathbf{x}_j}{\|\mathbf{x}_i\|_2 \|\mathbf{x}_j\|_2} = \frac{\sum_{d=1}^D x_{id} x_{jd}}{\sqrt{\sum_{d=1}^D x_{id}^2} \sqrt{\sum_{d=1}^D x_{jd}^2}}
\end{equation}
Giá trị đo lường này chuẩn hóa biến thiên trong đoạn $[-1, 1]$, hoàn toàn độc lập với độ dài hay quy mô tuyệt đối của các vectơ quan sát. Do tính chất phản ánh cấu trúc góc kẹp, độ đo này đặc biệt phù hợp và phát huy hiệu quả cao trong các bài toán xử lý dữ liệu thưa, khai phá dữ liệu văn bản, hoặc các hệ thống khuyến nghị, nơi tần suất xuất hiện tương đối mang nhiều tín hiệu hành vi hơn là giá trị thô cụ thể. Trong nghiên cứu phân khúc khách hàng dựa trên giao dịch tài chính này, độ tương đồng Cosine không được sử dụng trực tiếp để huấn luyện mô hình, nhưng đóng vai trò như một hệ quy chiếu thực nghiệm để so sánh các bản chất không gian hình học khác nhau.

\subsection{Lựa chọn độ đo trong thực tiễn}
\label{subsec:metric_selection_practice}

Trong ngữ cảnh thực thi bài toán phân cụm hành vi sử dụng thẻ tín dụng, khoảng cách Euclidean là lựa chọn tối ưu và mang tính hệ thống nhất dựa trên cấu trúc bản sinh của dữ liệu tài chính. Bộ dữ liệu nghiên cứu tồn tại dưới cấu trúc dạng bảng (tabular data), trong đó các thuộc tính biểu diễn dưới dạng các biến số liên tục phản ánh số dư, tổng lượng tiền giao dịch, hay các tỷ lệ tính năng đã qua tối ưu hóa.

Việc áp dụng khoảng cách Euclidean trong pipeline xử lý này hoàn toàn được bảo vệ vững chắc về mặt lý thuyết nhờ các luận điểm sau:
\begin{itemize}
    \item Toàn bộ các biến số liên tục có đuôi nặng đã được xử lý bất đối xứng thông qua phép biến đổi log, đưa phân phối về trạng thái tiệm cận phân phối chuẩn Gauss.
    \item Dữ liệu sau biến đổi tiếp tục được chuyển qua bộ chuẩn hóa quy mô Standard Scaler, triệt tiêu hoàn toàn sự chênh lệch về đơn vị đo lường vật lý và đưa phương sai của mỗi chiều đặc trưng về trạng thái đồng nhất bằng một.
\end{itemize}
Sau khi các bước tiền xử lý toán học trên được hoàn tất, không gian đặc trưng đã thỏa mãn chính xác các giả thiết hình học và xác suất thống kê của thuật toán K-Means lẫn mô hình Gauss hỗn hợp dạng hiệp phương sai đường chéo. Các cụm mật độ lúc này tồn tại dưới dạng các khối cầu lồi đồng đều cấu trúc không gian, cho phép khoảng cách Euclidean bóc tách một cách chính xác các ranh giới hành vi mà không làm mất mát hay méo mó thông tin của hệ thống.


\section{Thuật toán K-Means}
\label{sec:kmeans_algorithm}

Thuật toán K-Means là một trong những phương pháp phân cụm phân hoạch cứng kinh điển và được áp dụng phổ biến nhất trong lĩnh vực học máy không giám sát \cite{macqueen1967some}. Phương pháp này tiếp cận bài toán phân cụm dưới góc nhìn hình học không gian, tìm cách phân chia dữ liệu thành các khối cầu lồi độc lập dựa trên khoảng cách đến các tâm đại diện.

\subsection{Bài toán tối ưu hóa}
\label{subsec:kmeans_optimization}

Về mặt toán học hình thức, cho trước tập dữ liệu $\mathcal{D} = \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N\}$ gồm $N$ quan sát trong không gian số thực $D$ chiều $\mathbb{R}^D$. Mục tiêu của thuật toán K-Means là tìm ra một phân hoạch không gian tạo thành $K$ cụm $\mathcal{C} = \{C_1, C_2, \dots, C_K\}$ độc lập sao cho tổng bình phương khoảng cách Euclidean từ mỗi điểm dữ liệu đến tâm cụm tương ứng đạt giá trị nhỏ nhất.

Mỗi cụm $C_k$ được đặc trưng bởi một vectơ trọng tâm hình học $\mathbf{m}_k \in \mathbb{R}^D$. Hàm mục tiêu tối ưu hóa của K-Means, còn được gọi là hàm tổng sai số bình phương trong nội bộ các cụm, được thiết lập dưới cấu trúc toán học sau:
\begin{equation}
J(\mathcal{C}, \{\mathbf{m}_k\}_{k=1}^K) = \sum_{k=1}^K \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \mathbf{m}_k\|_2^2
\end{equation}
Bài toán phân cụm K-Means bản chất là một bài toán quy hoạch tổ hợp, tìm kiếm phân hoạch tối ưu $\mathcal{C}^*$ và tập hợp các tâm cụm tương ứng nhằm cực tiểu hóa hàm mục tiêu:
\begin{equation}
\label{eq:kmeans_argmin}
\mathcal{C}^*, \{\mathbf{m}_k^*\} = \arg\min_{\mathcal{C}, \mathbf{m}} J(\mathcal{C}, \{\mathbf{m}_k\}_{k=1}^K)
\end{equation}
Việc giải bài toán tối ưu hóa toàn cục trong Phương trình \ref{eq:kmeans_argmin} là một bài toán NP-hard ngay cả khi số lượng cụm $K = 2$ hoặc không gian chỉ có hai chiều. Do số lượng cấu hình phân hoạch không gian tăng theo hàm mũ của quy mô dữ liệu, việc duyệt cạn toàn bộ không gian nghiệm là bất khả thi về mặt tính toán thực tế. Do đó, các thuật toán xấp xỉ lặp bước đã được phát triển để tìm kiếm các điểm tối ưu cục bộ.

\subsection{Thuật toán Lloyd}
\label{subsec:lloyd_algorithm}

Thuật toán Lloyd là cơ chế phổ biến nhất được sử dụng để tìm kiếm nghiệm xấp xỉ cho bài toán K-Means thông qua một quy trình tối ưu hóa luân phiên lặp đi lặp lại. Thuật toán bắt đầu bằng việc thiết lập ngẫu nhiên $K$ tâm cụm ban đầu và thực thi tuần hoàn hai bước chiến lược sau cho đến khi đạt trạng thái hội tụ:

\begin{enumerate}
    \item Bước gán nhãn cụm : Cố định tập hợp các trọng tâm hình học $\{\mathbf{m}_k\}$, tiến hành phân phối từng điểm dữ liệu $\mathbf{x}_i$ vào cụm có tâm gần nó nhất theo khoảng cách Euclidean:
    \begin{equation}
    C_k^{(t)} = \left\{ \mathbf{x}_i : \|\mathbf{x}_i - \mathbf{m}_k^{(t)}\|_2^2 \leq \|\mathbf{x}_i - \mathbf{m}_j^{(t)}\|_2^2 \quad \forall j = 1, \dots, K \right\}
    \end{equation}
    
    \item Bước cập nhật tâm cụm : Cố định cấu trúc phân hoạch các cụm vừa tìm được ở bước trước, tiến hành tính toán lại tọa độ vị trí của các trọng tâm mới bằng cách lấy trung bình cộng tọa độ của tất cả các phần tử nằm trong cụm đó:
    \[
    \mathbf{m}_k^{(t+1)} = \frac{1}{|C_k^{(t)}|} \sum_{\mathbf{x}_i \in C_k^{(t)}} \mathbf{x}_i
    \]
\end{enumerate}

Quy trình lặp luân phiên này được đảm bảo chắc chắn sẽ hội tụ về một điểm tối ưu cục bộ do hàm mục tiêu $J$ là một hàm giảm đơn điệu và bị chặn dưới bởi giá trị không. Thuật toán sẽ dừng lại khi sự thay đổi của hàm mục tiêu giữa hai bước liên tiếp nhỏ hơn một ngưỡng quy định, hoặc khi nhãn phân cụm của toàn bộ các quan sát không còn sự biến động.

\subsection{Khởi tạo K-Means++}
\label{subsec:kmeans_plusplus}

Mặc dù thuật toán Lloyd có tốc độ tính toán rất nhanh, hiệu năng cuối cùng của nó lại phụ thuộc nặng nề vào cấu hình phân bổ của các tâm cụm khởi tạo ban đầu. Nếu các tâm cụm ban đầu được chọn quá gần nhau một cách ngẫu nhiên, thuật toán rất dễ bị rơi vào các bẫy tối ưu cục bộ kém chất lượng, dẫn đến kết quả phân hoạch sai lệch hoàn toàn so với cấu trúc mật độ thực tế.

Để giải quyết triệt để điểm yếu này, phương pháp khởi tạo K-Means++ đã được phát triển nhằm phân bổ các tâm cụm ban đầu cách xa nhau trong không gian đặc trưng thông qua một cơ chế phân phối xác suất \cite{arthur2006k}. Quy trình khởi tạo của K-Means++ được thực hiện tuần tự như sau:
\begin{itemize}
    \item Chọn tâm cụm đầu tiên $\mathbf{m}_1$ một cách ngẫu nhiên đồng đều từ tập dữ liệu $\mathcal{D}$.
    \item Đối với mỗi điểm dữ liệu $\mathbf{x}_i$, tính toán khoảng cách ngắn nhất từ nó đến tâm cụm gần nhất đã được chọn, ký hiệu là $D(\mathbf{x}_i)$.
    \item Lựa chọn tâm cụm tiếp theo $\mathbf{m}_k$ từ tập dữ liệu dựa trên một phân phối xác suất có trọng số tỉ lệ thuận với bình phương khoảng cách. Xác suất để một điểm $\mathbf{x}_i$ được chọn làm tâm cụm mới tuân theo công thức:
    \begin{equation}
    P(\mathbf{x}_i) = \frac{D(\mathbf{x}_i)^2}{\sum_{\mathbf{x}_j \in \mathcal{D}} D(\mathbf{x}_j)^2}
    \end{equation}
    \item Lặp lại quy trình tính toán khoảng cách và chọn tâm theo xác suất cho đến khi thu thập đủ $K$ tâm cụm ban đầu.
\end{itemize}
Cơ chế này đảm bảo các điểm nằm ở vùng biên, cách xa các tâm cụm hiện tại sẽ có cơ hội cao hơn được chọn làm tâm mới, từ đó giúp thuật toán Lloyd tăng tốc độ hội tụ và cải thiện đáng kể độ bền vững của nghiệm tối ưu cuối cùng.

\subsection{Lựa chọn số cụm K — Elbow Method}
\label{subsec:elbow_method}

Một trong những thách thức lớn nhất khi áp dụng K-Means trong thực tế là tham số số lượng cụm $K$ phải được xác định trước khi huấn luyện mô hình. Phương pháp khuỷu tay (Elbow Method) là một kỹ thuật đồ thị trực quan phổ biến dựa trên việc quan sát biến thiên của hàm mục tiêu WCSS để tìm ra số lượng cụm phù hợp.

Khi tăng giá trị $K$ từ $1$ lên các giá trị lớn hơn, giá trị hàm WCSS chắc chắn sẽ giảm dần do không gian được chia nhỏ và các điểm dữ liệu nằm gần tâm của chúng hơn. Trên đồ thị biểu diễn mối quan hệ giữa số lượng cụm $K$ và WCSS, tốc độ giảm của WCSS ban đầu sẽ rất nhanh tại các giá trị $K$ nhỏ, sau đó sẽ chậm lại rõ rệt khi vượt qua một ngưỡng $K$ tối ưu nhất định, tạo thành một điểm gãy khúc có hình dáng giống như một chiếc khuỷu tay. Về mặt lý thuyết đồ thị, điểm gãy khúc này đánh dấu ranh giới mà tại đó việc thêm cụm mới không còn mang lại nhiều giá trị thông tin để giải thích cho phương sai hình học của dữ liệu, biểu thị số lượng cụm tự nhiên ẩn chứa bên trong tập dữ liệu.

\subsection{Các giả thiết của mô hình}
\label{subsec:kmeans_assumptions}

Để thuật toán K-Means hoạt động hiệu quả và đạt được cấu trúc phân hoạch chính xác, dữ liệu đầu vào cần phải thỏa mãn các giả thiết hình học không gian ngầm định sau:
\begin{itemize}
    \item Cấu trúc hình cầu của các cụm: K-Means ngầm định các cụm dữ liệu có dạng hình khối cầu lồi đồng đều trong không gian đa chiều. Giả thiết này xuất phát trực tiếp từ việc sử dụng khoảng cách Euclidean làm độ đo tối ưu, khiến thuật toán hoạt động kém hiệu quả trên các tập dữ liệu có cấu trúc cụm kéo dài, dạng dải uốn lượn hoặc có hình học phi tuyến phức tạp.
    \item Quy mô và mật độ đồng nhất: Mô hình giả định các cụm có quy mô không gian và mật độ phân bổ điểm dữ liệu tương đương nhau. Nếu tồn tại các cụm có kích thước quá lớn nằm cạnh các cụm có kích thước cực nhỏ, bước gán nhãn khoảng cách Euclidean của K-Means sẽ có xu hướng xâm lấn, cắt đôi cụm lớn để gán bớt phần tử cho cụm nhỏ.
    \item Sự độc lập của các chiều đặc trưng: Việc đo lường khoảng cách trên tất cả các trục tọa độ với trọng số ngang nhau ngầm định các biến đặc trưng có mức độ quan trọng như nhau và không có sự tương quan tuyến tính quá chặt chẽ. Do đó, mô hình đòi hỏi dữ liệu phải được chuẩn hóa quy mô phương sai một cách cẩn thận trước khi huấn luyện để tránh hiện tượng các biến có miền giá trị lớn lấn át hoàn toàn các thuộc tính khác.
\end{itemize}


% ============================================================
%  Mục 2.5 — Mô hình hỗn hợp Gaussian (GMM)
%  File: chuong_2_section_2_5.tex
% ============================================================

\section{Mô hình hỗn hợp Gaussian}
\label{sec:gmm}


Thuật toán K-Means phân chia
không gian dữ liệu thành các vùng Voronoi cứng nhắc: mỗi điểm dữ liệu
thuộc về đúng một cụm duy nhất, không có sự không chắc chắn.
Cách tiếp cận này hiệu quả khi các cụm có dạng hình cầu, kích thước
tương đương và ranh giới tách biệt rõ ràng.
Tuy nhiên, trong thực tế, dữ liệu thường thể hiện các cấu trúc phức tạp
hơn: các cụm có thể có hình dạng ellipse, mật độ khác nhau, hoặc các
điểm nằm gần ranh giới có khả năng thuộc về nhiều cụm với mức độ tin cậy
khác nhau \cite{bishop2006pattern}.

Mô hình hỗn hợp Gaussian (Gaussian Mixture Model, GMM) giải quyết những
hạn chế này bằng cách đặt bài toán phân cụm trong khuôn khổ xác suất.
Thay vì gán nhãn cứng, GMM mô hình hóa phân phối xác suất của toàn bộ
tập dữ liệu như một hỗn hợp của $K$ phân phối Gaussian thành phần, mỗi
thành phần tương ứng với một cụm tiềm ẩn.
Kết quả là mỗi điểm dữ liệu được mô tả bởi một véc-tơ xác suất hậu
nghiệm (posterior probability) — còn gọi là trách nhiệm}(responsibility)
— thể hiện mức độ thuộc về từng cụm.
Đây được gọi là phân cụm mềm, đối lập với phân
cụm cứng của K-Means \cite{reynolds2009gaussian}.

% ────────────────────────────────────────────────────────────
\subsection{Mô hình xác suất sinh dữ liệu}
\label{subsec:gmm_model}

GMM giả định rằng tập dữ liệu $\mathcal{X} = \{\mathbf{x}_1, \mathbf{x}_2,
\ldots, \mathbf{x}_n\}$ với $\mathbf{x}_i \in \mathbb{R}^d$ được sinh ra
từ một phân phối hỗn hợp có dạng:

\begin{equation}
    p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mathbf{x} \mid
    \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
    \label{eq:gmm_density}
\end{equation}

\noindent trong đó:
\begin{itemize}
    \item $K$ là số thành phần Gaussian hay số cụm ,
    \item $\pi_k \in (0, 1)$ là trọng số hỗn hợp 
          của thành phần thứ $k$, thỏa mãn $\sum_{k=1}^{K} \pi_k = 1$,
    \item $\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k,
          \boldsymbol{\Sigma}_k)$ là hàm mật độ xác suất của phân phối
          Gaussian $d$-chiều với véc-tơ kỳ vọng $\boldsymbol{\mu}_k \in
          \mathbb{R}^d$ và ma trận hiệp phương sai $\boldsymbol{\Sigma}_k
          \in \mathbb{R}^{d \times d}$.
\end{itemize}

Hàm mật độ của phân phối Gaussian $d$-chiều được định nghĩa như sau:

\begin{equation}
    \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
    = \frac{1}{(2\pi)^{d/2} |\boldsymbol{\Sigma}_k|^{1/2}}
      \exp\!\left( -\frac{1}{2}
      (\mathbf{x} - \boldsymbol{\mu}_k)^\top
      \boldsymbol{\Sigma}_k^{-1}
      (\mathbf{x} - \boldsymbol{\mu}_k)
      \right)
    \label{eq:gaussian_pdf}
\end{parameter>

Tập tham số của mô hình cần được ước lượng từ dữ liệu là:
$\boldsymbol{\theta} = \{ \pi_k, \boldsymbol{\mu}_k,
\boldsymbol{\Sigma}_k \}_{k=1}^{K}$.
Phương pháp ước lượng phổ biến nhất là ước lượng hợp lý cực đại
(Maximum Likelihood Estimation, MLE), tức là tìm
$\boldsymbol{\theta}^*$ cực đại hóa log-likelihood của dữ liệu quan sát:

\begin{equation}
    \boldsymbol{\theta}^* = \arg\max_{\boldsymbol{\theta}}
    \sum_{i=1}^{n} \log \left[
    \sum_{k=1}^{K} \pi_k \,
    \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
    \right]
    \label{eq:gmm_loglik}
\end{equation}

Do hàm log-likelihood trong phương trình~\eqref{eq:gmm_loglik} chứa
logarithm của tổng, bài toán tối ưu hóa này không có nghiệm dạng đóng và phải được giải bằng phương pháp lặp, cụ thể là
thuật toán Expectation-Maximization (EM).

% ────────────────────────────────────────────────────────────
\subsection{Biến ẩn và ý nghĩa của phân cụm mềm}
\label{subsec:gmm_latent}

Để hiểu bản chất xác suất của GMM, ta giới thiệu biến ẩn
(latent variable) $z_i \in \{1, 2, \ldots, K\}$ biểu thị thành phần
Gaussian đã sinh ra điểm $\mathbf{x}_i$.
Biến $z_i$ không quan sát được trực tiếp — đây chính là lý do nó được
gọi là biến ẩn \cite{bishop2006pattern}.

Theo quy tắc Bayes, xác suất hậu nghiệm của biến ẩn — tức xác suất để
điểm $\mathbf{x}_i$ thuộc về thành phần $k$ — được tính như sau:

\begin{equation}
    r_{ik} = p(z_i = k \mid \mathbf{x}_i, \boldsymbol{\theta})
    = \frac{\pi_k \, \mathcal{N}(\mathbf{x}_i \mid
      \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}
      {\displaystyle\sum_{j=1}^{K} \pi_j \,
      \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j,
      \boldsymbol{\Sigma}_j)}
    \label{eq:responsibility}
\end{equation}

Đại lượng $r_{ik}$ được gọi là trách nhiệm của
thành phần $k$ đối với điểm $\mathbf{x}_i$, với tính chất $r_{ik} \in
[0, 1]$ và $\sum_{k=1}^{K} r_{ik} = 1$ với mọi $i$.

Đây chính là điểm khác biệt cơ bản so với K-Means: thay vì nhãn cứng
$z_i \in \{1, \ldots, K\}$, GMM cung cấp một phân phối xác suất đầy đủ
trên tập nhãn cho mỗi điểm dữ liệu.
Nhãn cụm cuối cùng (phục vụ mục đích phân tích) thường được xác định
bằng quy tắc hậu nghiệm cực đại:

\begin{equation}
    \hat{z}_i = \arg\max_{k} \; r_{ik}
    \label{eq:map_assignment}
\end{equation}

% ────────────────────────────────────────────────────────────
\subsection{Thuật toán EM cho GMM}
\label{subsec:gmm_em}

Thuật toán Expectation-Maximization (EM) là phương
pháp chuẩn để ước lượng tham số của GMM.
EM hoạt động theo cơ chế lặp xen kẽ giữa hai bước: bước E
(Expectation) và bước M (Maximization), đảm bảo log-likelihood
không giảm qua mỗi vòng lặp \cite{wu1983convergence}.

Khởi tạo.
Tham số ban đầu $\boldsymbol{\theta}^{(0)}$ được khởi tạo, thường bằng
cách sử dụng kết quả của K-Means++ để có điểm khởi đầu ổn định hơn so
với khởi tạo ngẫu nhiên thuần túy \cite{scikit-learn}.

Bước E (Expectation).
Với tham số hiện tại $\boldsymbol{\theta}^{(t)}$, tính trách nhiệm
$r_{ik}^{(t)}$ cho tất cả các cặp $(i, k)$ theo
phương trình~\eqref{eq:responsibility}.
Bước này tương đương với tính kỳ vọng của log-likelihood đầy đủ
(complete-data log-likelihood) theo phân phối hậu nghiệm của biến ẩn.

Bước M (Maximization).
Cập nhật tham số bằng cách cực đại hóa kỳ vọng vừa tính được.
Nghiệm dạng đóng cho từng tham số là:

\begin{align}
    N_k^{(t)} &= \sum_{i=1}^{n} r_{ik}^{(t)}
    \label{eq:em_Nk} \\[4pt]
    \boldsymbol{\mu}_k^{(t+1)} &= \frac{1}{N_k^{(t)}}
    \sum_{i=1}^{n} r_{ik}^{(t)} \, \mathbf{x}_i
    \label{eq:em_mu} \\[4pt]
    \boldsymbol{\Sigma}_k^{(t+1)} &= \frac{1}{N_k^{(t)}}
    \sum_{i=1}^{n} r_{ik}^{(t)}
    (\mathbf{x}_i - \boldsymbol{\mu}_k^{(t+1)})
    (\mathbf{x}_i - \boldsymbol{\mu}_k^{(t+1)})^\top
    \label{eq:em_sigma} \\[4pt]
    \pi_k^{(t+1)} &= \frac{N_k^{(t)}}{n}
    \label{eq:em_pi}
\end{align}

Điều kiện dừng.
Hai bước E và M được lặp lại cho đến khi sự thay đổi của log-likelihood
giữa hai vòng lặp liên tiếp nhỏ hơn một ngưỡng $\epsilon$ cho trước,
hoặc đạt số vòng lặp tối đa.

Độ phức tạp tính toán.
Mỗi vòng lặp EM có độ phức tạp $O(nKd^2)$ do phải tính ma trận hiệp
phương sai $d \times d$ cho mỗi thành phần.
So với K-Means có độ phức tạp $O(nKd)$ mỗi vòng lặp, EM tốn kém hơn
đáng kể khi $d$ lớn, đặc biệt với kiểu hiệp phương sai \texttt{full}
\cite{bishop2006pattern}.

% ────────────────────────────────────────────────────────────
\subsection{Các dạng ma trận hiệp phương sai}
\label{subsec:gmm_covariance}

Một điểm mạnh của GMM là tính linh hoạt trong việc lựa chọn cấu trúc
của ma trận hiệp phương sai $\boldsymbol{\Sigma}_k$.
Mỗi lựa chọn tương ứng với một giả thiết khác nhau về hình dạng hình học
của các cụm, đồng thời ảnh hưởng trực tiếp đến số lượng tham số cần ước
lượng và nguy cơ overfitting \cite{mclachlan2000finite}.
Bốn kiểu phổ biến được sử dụng trong thực tiễn là:

\begin{itemize}
    \item Full: $\boldsymbol{\Sigma}_k$ là ma trận hiệp phương
    sai đầy đủ, không có ràng buộc cấu trúc.
    Mỗi cụm có thể có hình dạng ellipse tùy ý, xoay theo mọi hướng.
    Đây là kiểu linh hoạt nhất nhưng cũng tốn nhiều tham số nhất:
    $O(Kd^2)$ tham số.

    \item Tied: Tất cả các thành phần chia sẻ cùng một ma trận
    hiệp phương sai $\boldsymbol{\Sigma}_k = \boldsymbol{\Sigma}$ với
    mọi $k$.
    Giảm số tham số xuống $O(d^2)$ nhưng áp đặt ràng buộc mạnh về hình
    dạng đồng nhất của các cụm.

    \item Diag: $\boldsymbol{\Sigma}_k$ là ma trận đường chéo,
    tức các chiều đặc trưng được giả thiết là độc lập có điều kiện.
    Các cụm có dạng ellipse với trục song song với trục tọa độ.
    Số tham số là $O(Kd)$.

    \item Spherical: $\boldsymbol{\Sigma}_k = \sigma_k^2
    \mathbf{I}$, tức mỗi cụm là một khối cầu với phương sai đồng nhất
    trên tất cả các chiều.
    Đây là trường hợp gần nhất với K-Means về mặt hình học, với chỉ $K$
    tham số cho toàn bộ cấu trúc phương sai.
\end{itemize}

Bảng~\ref{tab:gmm_cov} tóm tắt đặc điểm của bốn kiểu hiệp phương sai.

\begin{table}[ht]
    \centering
    \caption{So sánh các kiểu ma trận hiệp phương sai trong GMM}
    \label{tab:gmm_cov}
    \begin{tabular}{lcccc}
        \toprule
        \textbf{Kiểu} & \textbf{Hình dạng cụm} &
        \textbf{Số tham số}  \\
        \midrule
        Full       & Ellipse tùy ý          & $O(Kd^2)$     \\
        Tied       & Ellipse đồng nhất      & $O(d^2)$    \\
        Diag       & Ellipse căn chỉnh trục & $O(Kd)$     \\
        Spherical  & Hình cầu               & $O(K)$       \\
        \bottomrule
    \end{tabular}
\end{table}

Việc lựa chọn kiểu hiệp phương sai phù hợp thường được thực hiện thông
qua các tiêu chí lựa chọn mô hình như BIC và AIC, được trình bày ở
mục~\ref{subsec:gmm_bic}.

% ────────────────────────────────────────────────────────────
\subsection{Lựa chọn số cụm \texorpdfstring{$K$}{K} --- Tiêu chí BIC và AIC}
\label{subsec:gmm_bic}

Khác với K-Means, GMM cung cấp một cơ chế lựa chọn mô hình tự nhiên dựa
trên nền tảng xác suất.
Hai tiêu chí phổ biến nhất là Bayesian Information Criterion (BIC)
và Akaike Information Criterion (AIC), đều cân bằng giữa khả năng
khớp dữ liệu (log-likelihood) và độ phức tạp của mô hình (số tham số)
\cite{akaike1974new, schwarz1978estimating}.

Akaike Information Criterion (AIC).
AIC được định nghĩa là:
\begin{equation}
    \text{AIC} = 2p - 2\hat{\ell}
    \label{eq:aic}
\end{equation}
trong đó $p$ là số tham số tự do của mô hình và $\hat{\ell}$ là giá trị
log-likelihood cực đại.
Số tham số của GMM phụ thuộc vào kiểu hiệp phương sai; ví dụ với kiểu
\texttt{full}: $p = K \cdot d \cdot (d+1)/2 + Kd + (K-1)$.
AIC ưu tiên mô hình có log-likelihood cao nhưng phạt nhẹ hơn với số tham
số lớn, do đó có xu hướng chọn $K$ lớn hơn so với BIC \cite{akaike1974new}.

Bayesian Information Criterion (BIC).
BIC được định nghĩa là:
\begin{equation}
    \text{BIC} = p \ln n - 2\hat{\ell}
    \label{eq:bic}
\end{equation}
So với AIC, BIC phạt mạnh hơn đối với số tham số khi cỡ mẫu $n$ lớn
(do hệ số $\ln n > 2$ khi $n > 7$).
BIC có nền tảng Bayesian vững chắc hơn và thường được khuyến nghị trong
các bài toán phân cụm vì tính nhất quán: trong điều kiện
phù hợp, BIC hội tụ về số thành phần đúng khi $n \to \infty$
\cite{schwarz1978estimating, fraley2002model}.


Mô hình với $K$ cho giá trị BIC (hoặc AIC) nhỏ nhất được coi là
tốt nhất.
Thực tế, ta thường huấn luyện GMM với nhiều giá trị $K$ và nhiều kiểu
hiệp phương sai, sau đó so sánh BIC/AIC để lựa chọn cấu hình tối ưu.
Trong trường hợp BIC và AIC cho kết quả khác nhau, BIC thường được ưu
tiên hơn vì xu hướng phạt mô hình phức tạp mạnh hơn, tránh overfitting
\cite{fraley2002model}.

% ────────────────────────────────────────────────────────────
\subsection{Giả thiết mô hình và điều kiện áp dụng}
\label{subsec:gmm_assumptions}

GMM hoạt động hiệu quả nhất khi các điều kiện sau được thỏa mãn, hoặc
xấp xỉ thỏa mãn trong thực tế:

\begin{enumerate}
    \item Giả thiết phân phối Gaussian.
    Mỗi cụm thực sự được sinh từ một phân phối Gaussian, hoặc ít nhất
    có dạng đơn đỉnh (unimodal) và xấp xỉ đối xứng.
    Khi dữ liệu có phân phối lệch nặng (heavy-tailed) hoặc đa đỉnh trong
    từng cụm, giả thiết này bị vi phạm và GMM có thể cần nhiều thành phần
    hơn để mô hình hóa một cụm thực sự.

    \item Số thành phần $K$ được xác định trước.
    Tương tự K-Means, GMM đòi hỏi $K$ là tham số đầu vào.
    Mặc dù BIC/AIC cung cấp cơ chế lựa chọn $K$ có cơ sở lý thuyết,
    việc quét toàn bộ không gian $K$ vẫn là cần thiết và tốn kém về tính
    toán.

    \item Nguy cơ suy biến.
    Khi một thành phần Gaussian chỉ chịu trách nhiệm cho rất ít điểm dữ
    liệu, ma trận hiệp phương sai có thể trở nên kỳ dị (singular), khiến
    log-likelihood tiến ra vô cực.
    Điều này thường được xử lý bằng cách thêm hằng số nhỏ vào đường chéo
    của $\boldsymbol{\Sigma}_k$ (regularization) \cite{scikit-learn}.

    \item Nhạy cảm với khởi tạo.
    EM chỉ đảm bảo hội tụ về cực trị cục bộ của log-likelihood, không
    nhất thiết là cực trị toàn cục \cite{wu1983convergence}.
    Do đó, kết quả phụ thuộc vào điểm khởi tạo.
    Trong thực tiễn, GMM thường được chạy nhiều lần với các điểm khởi
    tạo khác nhau (\texttt{n\_init} lần) và kết quả tốt nhất theo
    log-likelihood được chọn.

    \item Yêu cầu về cỡ mẫu.
    Với kiểu hiệp phương sai \texttt{full}, số tham số tăng theo $O(Kd^2)$.
    Khi $d$ lớn và $n$ không đủ lớn, mô hình dễ bị overfitting.
    Kiểu \texttt{diag} hoặc \texttt{spherical} nên được cân nhắc trong
    trường hợp này.
\end{enumerate}

\section{Đánh giá chất lượng phân cụm}
\label{sec:clustering_evaluation}

Sau khi cấu trúc phân hoạch dữ liệu được thiết lập bởi các thuật toán như K-Means hoặc mô hình hỗn hợp Gaussian, hệ thống đòi hỏi một cơ chế kiểm định độc lập để đánh giá chất lượng và độ bền vững của các phân khúc khách hàng thu được. Giai đoạn này đóng vai trò quyết định để minh chứng cho tính hợp lý của mô hình trước khi chuyển sang bước phác họa hồ sơ hành vi.

\subsection{Tổng quan — tại sao không dùng external metrics?}
\label{subsec:evaluation_overview}

Trong lý thuyết học máy phân cụm, các độ đo đánh giá hiệu năng mô hình được chia thành hai trường phái tiếp cận chính: các độ đo ngoại vi (External Metrics) và các độ đo nội bộ (Internal Validation Metrics).
\begin{itemize}
    \item Bản chất của độ đo ngoại vi: Các chỉ số ngoại vi như độ chính xác phân lớp, chỉ số Rand hiệu chỉnh hay độ tương hỗ thông tin chuẩn hóa đòi hỏi sự tồn tại của một tập nhãn mục tiêu chuẩn do con người gán trước làm hệ quy chiếu để đối chiếu trực tiếp kết quả phân cụm.
    \item Lý do không sử dụng độ đo ngoại vi trong bài toán thực tế: Bản chất cốt lõi của bài toán phân khúc hành vi khách hàng sử dụng thẻ tín dụng là một bài toán khám phá cấu trúc ẩn, hoàn toàn không tồn tại bất kỳ một tập nhãn chuẩn ngoại vi nào về các nhóm khách hàng. Nếu nhãn chuẩn đã tồn tại, bài toán sẽ suy biến thành học có giám sát. Do đó, việc sử dụng các độ đo ngoại vi trong ngữ cảnh này là bất khả thi và sai lệch về mặt phương pháp luận nghiên cứu không giám sát.
\end{itemize}

Do thiếu vắng các thông tin định hướng ngoại vi, việc xác thực hiệu năng bắt buộc phải dựa vào các tiêu chí đánh giá nội bộ. Các độ đo này khai thác trực tiếp cấu trúc khoảng cách hình học và ma trận phân tán của các điểm dữ liệu trong không gian đặc trưng đã chuẩn hóa nhằm định lượng trạng thái tối ưu của phân hoạch mà không cần bất kỳ thông tin bổ trợ nào ngoài tập dữ liệu gốc.



% ------------------------------------------------------------
\subsection{Silhouette Score}
\label{subsec:silhouette}
% ------------------------------------------------------------

Silhouette Score là chỉ số đánh giá phân cụm
\cite{rousseeuw1987}, kết hợp cùng lúc hai tiêu chí gắn kết trong cụm và tách biệt giữa các cụm thành một giá trị đơn dễ diễn giải.

\subsubsection*{Định nghĩa và công thức}

Với điểm dữ liệu $x_i$ thuộc cụm $C_p$, hai đại lượng được xác định:

Độ gắn kết $a(i)$: trung bình khoảng cách từ $x_i$ đến tất cả các
điểm còn lại trong cùng cụm $C_p$:

\begin{equation}
    a(i) = \frac{1}{|C_p| - 1} \sum_{\substack{j \in C_p \\ j \neq i}} d(x_i, x_j)
    \label{eq:silhouette_a}
\end{equation}

Giá trị $a(i)$ nhỏ biểu thị $x_i$ nằm gần trung tâm cụm của mình, tức cụm có độ gắn
kết cao.

Độ tách biệt $b(i)$: trung bình khoảng cách nhỏ nhất từ $x_i$ đến
các điểm thuộc cụm lân cận gần nhất $C_q$ ($q \neq p$):

\begin{equation}
    b(i) = \min_{q \neq p} \frac{1}{|C_q|} \sum_{j \in C_q} d(x_i, x_j)
    \label{eq:silhouette_b}
\end{equation}

Giá trị $b(i)$ lớn biểu thị $x_i$ nằm xa các cụm khác, tức các cụm được tách biệt
tốt.

Hệ số silhouette của điểm $x_i$ được định nghĩa là:

\begin{equation}
    s(i) = \frac{b(i) - a(i)}{\max\bigl(a(i),\; b(i)\bigr)}
    \label{eq:silhouette_si}
\end{equation}

Từ công thức~\eqref{eq:silhouette_si}, rõ ràng $s(i) \in [-1, 1]$:

\begin{itemize}
    \item $s(i) \approx +1$: điểm $x_i$ được gán đúng cụm --- nó rất gần cụm của mình
    ($a(i)$ nhỏ) và rất xa cụm lân cận ($b(i)$ lớn).
    \item $s(i) \approx 0$: điểm $x_i$ nằm gần ranh giới giữa hai cụm, có thể thuộc
    về cụm này hoặc cụm kia.
    \item $s(i) < 0$: điểm $x_i$ bị gán sai cụm --- nó thực sự gần một cụm khác hơn
    cụm hiện tại của nó.
\end{itemize}

Silhouette Score tổng thể của toàn bộ phân cụm là trung bình cộng:

\begin{equation}
    \bar{s} = \frac{1}{n} \sum_{i=1}^{n} s(i)
    \label{eq:silhouette_avg}
\end{equation}


\subsubsection*{Ứng dụng chọn K}

Ngoài đánh giá chất lượng phân cụm cho một $K$ cụ thể, Silhouette Score còn được dùng
để chọn số cụm tối ưu bằng cách tính $\bar{s}(K)$ với nhiều giá trị $K$ và chọn $K$
cho $\bar{s}$ lớn nhất. Đây là phương pháp bổ sung cho Elbow Method, đặc biệt hữu ích
khi đồ thị Elbow không có điểm gãy rõ ràng.

% ------------------------------------------------------------
\subsection{Davies-Bouldin Index}
\label{subsec:davies_bouldin}
% ------------------------------------------------------------

Davies-Bouldin Index (DBI) được định nghĩa là
\cite{davies1979cluster} một chỉ số đánh giá mức độ tương đồng giữa các cụm, dựa
trên tỉ số giữa tán xạ trong cụm và khoảng cách tâm cụm.

\subsubsection*{Định nghĩa và công thức}

Với cụm $C_i$, độ tán xạ nội tại được đo bằng trung bình
khoảng cách từ các điểm đến tâm cụm:

\begin{equation}
    s_i = \frac{1}{|C_i|} \sum_{x \in C_i} d(x,\, \mu_i)
    \label{eq:dbi_scatter}
\end{equation}

trong đó $\mu_i$ là tâm của cụm $C_i$. Đại lượng này đo mức độ ``rộng'' của cụm: cụm
nhỏ gọn (compact) có $s_i$ nhỏ.

Độ tương đồng giữa hai cụm $C_i$ và $C_j$ được định nghĩa là tỉ số giữa tổng tán xạ
nội tại và khoảng cách giữa hai tâm cụm:

\begin{equation}
    R_{ij} = \frac{s_i + s_j}{d(\mu_i, \mu_j)}
    \label{eq:dbi_ratio}
\end{equation}

Cụm $C_i$ bị so sánh với cụm tương đồng nhất của nó:

\begin{equation}
    D_i = \max_{j \neq i}\; R_{ij}
    \label{eq:dbi_di}
\end{equation}

Davies-Bouldin Index là trung bình của $D_i$ trên tất cả các cụm:

\begin{equation}
    \mathrm{DBI} = \frac{1}{K} \sum_{i=1}^{K} D_i
    = \frac{1}{K} \sum_{i=1}^{K} \max_{j \neq i}
    \frac{s_i + s_j}{d(\mu_i, \mu_j)}
    \label{eq:dbi_final}
\end{equation}

\subsubsection*{Diễn giải và tính chất}

DBI càng thấp thì phân cụm càng tốt: giá trị thấp có nghĩa là các cụm vừa
nhỏ gọn vừa tách biệt xa nhau. DBI $= 0$ là trường hợp lý
tưởng tức các cụm hoàn toàn tách biệt, không chồng lấp. Không có ngưỡng tuyệt đối; DBI
thường được dùng để so sánh tương đối giữa các cấu hình phân cụm khác nhau.

Một đặc điểm đáng chú ý là DBI không phụ thuộc vào số cụm $K$ trong công thức, nên
có thể dùng trực tiếp để so sánh các phân cụm với $K$ khác nhau. Tuy nhiên, DBI có xu
hướng giảm khi $K$ tăng (vì các cụm nhỏ hơn thường nhỏ gọn hơn), nên cần cẩn thận
khi dùng DBI đơn lẻ để chọn $K$ \cite{davies1979cluster}.

% ------------------------------------------------------------
\subsection{Calinski-Harabasz Index}
\label{subsec:calinski_harabasz}
% ------------------------------------------------------------

Calinski-Harabasz Index (CHI), còn gọi là Variance Ratio Criterion (VRC), được
Caliński và Harabasz đề xuất \cite{calinski1974dendrite}. Chỉ số này tiếp
cận bài toán đánh giá phân cụm từ góc độ phân tích phương sai (ANOVA): một phân cụm
tốt là phân cụm trong đó phương sai giữa các cụm lớn so
với phương sai trong từng cụm.

\subsubsection*{Định nghĩa và công thức}

Ký hiệu $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$ là tâm toàn cục của tập dữ liệu.

Tổng bình phương giữa các cụm (Between-Cluster Sum of Squares --- BCSS) đo mức độ
tách biệt giữa các tâm cụm so với tâm toàn cục:

\begin{equation}
    \mathrm{BCSS} = \sum_{k=1}^{K} n_k \|\mu_k - \bar{x}\|^2
    \label{eq:bcss}
\end{equation}

trong đó $n_k = |C_k|$ là số điểm trong cụm $k$.

Tổng bình phương trong cụm (Within-Cluster Sum of Squares --- WCSS) đo mức độ gắn kết
nội tại --- đây chính là hàm mục tiêu của K-Means trong công thức~\eqref{eq:wcss_kmeans}:

\begin{equation}
    \mathrm{WCSS} = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2
    \label{eq:wcss_chi}
\end{equation}

Calinski-Harabasz Index là tỉ số phương sai giữa các cụm và trong cụm, được chuẩn hóa
theo bậc tự do:

\begin{equation}
    \mathrm{CHI}(K) = \frac{\mathrm{BCSS} / (K - 1)}{\mathrm{WCSS} / (n - K)}
    \label{eq:chi}
\end{equation}

Mẫu số $(K-1)$ và $(n-K)$ là bậc tự do tương ứng của BCSS và WCSS, tương tự thống kê
$F$ trong phân tích phương sai một chiều (one-way ANOVA). Vì vậy CHI còn được gọi là
pseudo-$F$ statistic.


\subsubsection*{Diễn giải và tính chất}

CHI càng cao thì phân cụm càng tốt: giá trị cao biểu thị phương sai giữa các
cụm lớn (tức các cụm tách biệt xa nhau) đồng thời phương sai trong cụm nhỏ (tức các
điểm gắn kết với tâm cụm).

Tuy nhiên, cần lưu ý một đặc điểm quan trọng: CHI có xu hướng tăng đơn điệu khi $K$
tăng do BCSS tăng còn WCSS giảm. Điều này khiến CHI không phù hợp để chọn $K$ thông
qua cực đại tuyệt đối --- thay vào đó, người ta thường tìm giá trị $K$ tại đó tốc độ
tăng của CHI chậm lại đáng kể (tương tự Elbow Method). CHI hoạt động tốt nhất khi
được dùng để so sánh các cấu hình phân cụm có cùng $K$ \cite{calinski1974dendrite}.





