---
title: "The Learning Problem part 1"
categories:
  - Learning from Data
tags:
  - Basic Machine Learning
  - Learning From Data
last_modified_at: 2018-04-15T12:54:35-05:00
---



Catatan pertama di blog. Yuhu :)

Catatan ini akan membahas pengenalan dasar Machine Learning. Fokus yang dibahas :
- Apa itu Machine Learning ?
- Mungkinkah sebuah mesin belajar ? Jika iya, bagaimana ?

Sebelum membahas Machine Learning, mari bayangkan sebuah eskperimen sederhana :

1. Jika kita menunjukan gambar angka 1-5 pada anak TK dan menanyakan gambar mana yang merupakan angka 4, 
maka kemungkinan besar anak tersebut akan memberikan jawaban yang benar.
{: .text-justify}

<figure class="align-center">
  <a href="#"><img src="{{ '/images/mnist.jpg' | absolute_url }}" alt=""></a>
  <figcaption>Angka 0-9 pada MNIST dataset</figcaption>
</figure> 

2. Jika kita menanyakan pada orang dewasa apa definisi atau aturan dari suatu gambar sehingga disebut 
gambar angka 4 (akan lebih mudah menentukan mana a dan b jika kita tahu definisi a dan b), maka 
kemungkinan besar orang tersebut tidak bisa memberikan jawaban yang jelas. 
{: .text-justify}


> **We didn't learn what a tree (or number) is by studying the mathematical definition of a trees.**
> We learned it by looking at trees. In other words we learned from 'data'.
> <footer><strong>Learning from Data</strong> &mdash;  </footer>
	
The best first pharagraph on Machine Learning books :)

Prinsip yang sama juga ingin kita terapkan pada sebuah mesin. Kita ingin sebuah mesin mengerjakan sesuatu 
bukan dari definisi(model atau set of instructions) yang kita masukkan, melainkan dari data yang kita berikan. 
Inilah intuisi dasar dari Machine Learning (frasa Machine Learning dipakai untuk membedakan denga proses 
Human Learning). 
{: .text-justify}

Pembeda Machine Learning dengan istilah Artificial Intellegence dan Deep Learning :
* AI : Konsep yang abstrak, lebih berkaitan dengan Cognitive Research. What is artificial ? and what is intellegence ?
* Deep Learning: Salah satu teknik (learning model) pada Machine Learning, Neural Net dengan hidden layer yang banyak/dalam (deep) 


<figure class="align-center">
  <a href="#"><img src="{{ '/images/ai-ml-dl.jpg' | absolute_url }}" alt=""></a>
  <figcaption>Artificial Intellegence, Machine learning, Deep Learning </figcaption>
</figure> 
 
## Contoh Machine Learning
 
Salah satu contoh penggunaan Machine Learning adalah untuk membangun recommender system. Hampir semua online 
platform yang ada memiliki recommender system
{: .text-justify}

* Facebook: "People You May Know"
* Netflix : "Other Movies You May Enjoy"
* LinkedIn: "Jobs You May Be Interested In"
* Twitter : "Trends for You"
* YouTube : "Recommended Videos"

Kita ambil contoh Netflix, recommender system untuk menentukan film2 yang akan direkomendasikan Netflix kepada user. 
Salah satu komponen yang penting adalah seberapa besar rating film tersebut (secara personal) menurut user. Netflix 
mempunyai data rating film yang diberikan oleh user yang sudah melihat film tersebut. Misal data tersebut memiliki 
format [Viewer, Film, Rating]. Dari data yang ada tersebut kita ingin memprediksi rating sebuah film yang belum 
pernah user lihat.
{: .text-justify}

Kita bisa membuat model empiris untuk memprediksi nilai rating berdasarkan preferensi viewer dan isi film tersebut. 
Untuk viewer bisa kita deskripsikan dalam vektor 
{: .text-justify}

* seberapa sering dia menonton genre komedi/aksi/drama/....
* seberapa sering dia menonton film berbahasa Inggris/Jepang/Korea/....
* seberapa sering dia menonton film-nya Nolan/Del Toro/Russo/....
* seberapa sering dia menonton film yang pemainnya ada Emma W/Emma Stone/Emma.....
* dst

Sedangkan dari sisi film kita bisa deskripsikan juga dalam vektor
* seberapa banyak unsur komedi/aksi/drama dalam film
* bahasa yang digunakan
* siapa director film tersebut
* siapa lead actor/actress di film tersebut
* dst

<figure class="align-center">
  <a href="#"><img src="{{ '/images/movie-rating-array.jpg' | absolute_url }}" alt=""></a>
  <figcaption>Learning from Data. Model untuk prediksi rating sebuah film</figcaption>
</figure> 

Dari dua vektor ini kita dapat melakukan dot product untuk menentukan prediksi nilai rating film tersebut menurut 
sang viewer. Namun pemilihan feature (genre, bahasa, sutradara, lead actor/actress) secara manual untuk membuat 
model seringkali menyita tenaga dan waktu pekerja (membuat kurang produktif).
{: .text-justify}

Dengan menggunakan Machine Learning(terutama Deep Learning) diharapkan proses pemilihan feature [User, Movie, Rating] 
dapat dihilangkan. Untuk melakukan hal tersebut Learning Algorithm akan melakukan 'reverse engineering' hanya 
berdasarkan nilai rating sebelumnya. 
{: .text-justify}

Berdasarkan input x (informasi viewer dan film), output y (nilai rating), maka data [Viewer, Film, Rating] dapat 
dijadikan dataset D yaitu pasangan (x1, y1),(x2, y2)...(xn, yn), X adalah input space (semua kemungkinan nilai x, 
kombinasi informasi viewer dan film) dan Y adalah output space (semua kemungkinan nilai y, untuk rating misal 
dari 0-100). Terdapat target function yang ingin dicari dan tidak diketahui y = f(x), dan kumpulan fungsi yang 
akan diuji yaitu H (Hypothesis set).
{: .text-justify}

<figure class="align-center">
  <a href="#"><img src="{{ '/images/learning-components.jpg' | absolute_url }}" alt=""></a>
  <figcaption>Learning from Data. Diagram komponen machine learning </figcaption>
</figure> 

### POINT x,y,X,Y,H,dan A (dalam Mathjax)

Target function (f(x)) dan data set (x1, y1),(x2, y2)...(xn, yn) merupakan hal yang melekat pada satu persoalan, 
tidak dapat diubah. tetapi learning algorithm dan set hipotesis  dapat kita ubah sesuai keinginan. Dua hal inilah 
yang akan menjadi tool kita untuk menyelesaikan persoalan. Learning algorithm dan set hipotesis sering kali digabung
menjadi Learning Model.
{: .text-justify}

Beberapa contoh Learning Model sederhana dan H (Hypothesis set): 
* Linear regression (H melingkupi semua polinomial orde 1)
* Decision Tree (H melingkupi a set of boolean)
* Multilayer perceptron tanpa aktivasi (H melingkupi persamaan polinomial orde n)

Termasuk pada Learning model yaitu menentukan bagaimana cara menentukan dan mengupdate koefisien pada persamaan 
hipotesis (gradient based, genetic algorithm) dan persamaan untuk menentukan error (SSE, Quadratic, RMSE, Cross Entropy dll) 
{: .text-justify}

**Note**: istilah learning algorithm dan learning model sering kali memiliki arti berbeda pada sumber lain.

Dari dataset D, Learning Algorithm akan memilih fungsi g dari Hypothesis set (H) yang paling mendekati target function 
(jumlah selisih antara f(y) dan g(y) paling kecil diantara semua set hipotesis yang ada)
{: .text-justify}

<figure class="align-center">
  <a href="#"><img src="{{ '/images/hypothesis-set.jpg' | absolute_url }}" alt=""></a>
  <figcaption>Machine Learning Tutorial. Ilustrasi pemilihan fungsi g dari Hypothesis set (H)  </figcaption>
</figure> 

Feature yang dihasilkan mungkin tidak seintuitif konten komedi atau aktor, bahkan bisa 
saja sangat abstrak. 
{: .text-justify}

> After all the algorithm is only trying to find the best way to predict how a viewer would rate a movie, not necessarily explain to us how it is done.
> <footer><strong>Learning from Data</strong> &mdash; </footer>
	
## Learning vs Design

Setelah membahas mengenai Learning kita kan membahas mengenai apa yang tidak termasuk learning. Pada bidang pattern 
recognition selain metode learning terdapat juga metode design. Metode learning berbasis kepada data (data-driven) 
sedangkan metode design berbasis pada spesifikasi atau  pengetahuan kita terhadap faktor intrinsik dari suatu 
permasalahan (domain-driven).  
{: .text-justify}

Misal : Kita ingin membuat sistem untuk menentukan nominal nilai sebuah koin pada vending machine.

Metode Learning
Kita mengumpulkan ukuran, berat dan nominal koin menjadi sebuah data set. Ukuran dan berat koin kita jadikan 
input vektor (x) dan nominal koin menjadi output vektor (y). Kemudian kita pilih learning model yang akan digunakan
(misal dengan K-means) sehingga terpilih hipotesis y = g(x) yang fit pada data set. Fungsi g(x) inilah yang nantinya
akan digunakan untuk menentukan nominal koin berdasar input yang masuk.
{: .text-justify}

<figure class="align-center">
  <a href="#"><img src="{{ '/images/coin-recognition-data.jpg' | absolute_url }}" alt=""></a>
  <figcaption>Learning from Data. (Kiri) Persebaran data ukuran, berat dan nominal koin. (Kanan) Sistem klasifikasi yang dihasilkan.  </figcaption>
</figure> 


Metode Design
Kita tanyakan spesifikasi uang logam kepada PERURI dan jumlah uang logam yang beredar kepada BI. Berdasarkan spesfikasi 
tersebut kita buat distribusi probabilitas berdasar ukuran, berat dan nominal. Nilai nominal nantinya akan ditentukan 
dari nilai probabilitas tertinggi pada input yang masuk. 
{: .text-justify}

<figure class="align-center">
  <a href="#"><img src="{{ '/images/coin-recognition-design.jpg' | absolute_url }}" alt=""></a>
  <figcaption>Learning from Data. (Kiri) Model probabilistik berdasarkan spesifkasi. (Kanan) Sistem klasifikasi yang dihasilkan.  </figcaption>
</figure> 

Perbedaan utama antara learning dan design adalah pada penggunaan data. Pada metode design kita dapat menentukan
fungsi target f berdasarkan model analitik (tanpa perlu melihat data) sedangkan pada metode learning kita memerlukan data
untuk menentukan fungsi target f. Kedua metode tentunya memiliki kelemahan dan keunggulan masing-masing.
{: .text-justify}

## Types of Learning

Proses learning berdasar data masih merupakan konsep yang sangat luas, oleh karena muncul beberapa paradigma learning 
begantung pada situasi permasalahan dan asumsi yang digunakan. Variasi paradigma umumnya bergantung pada jenis data 
set yang digunakan. Paradigma learning yang sudah dibahas (dan akan difokuskan pada seri catatan ini) disebut supervised 
learning.
{: .text-justify}

1. Supervised Learning 
Jika dataset mengandung label (output yang diharapkan). Paradigma yang paling banyak digunakan. 
Dataset : (input, correct output).

<figure class="align-center">
  <a href="#"><img src="{{ '/images/supervised-learning.jpg' | absolute_url }}" alt=""></a>
  <figcaption>Machine Learning Tutorial. (Kiri) Data set dengan label berupa warna . (Kanan) Sistem klasifikasi yang dihasilkan.  </figcaption>
</figure> 

2. Unsupervised Learning
Jika dataset hanya berisi nilai input saja, tidak ada label. Biasanya digunakan untuk clustering  atau factor analysis. 
Dataset : (input).

<figure class="align-center">
  <a href="#"><img src="{{ '/images/unsupervised-learning.jpg' | absolute_url }}" alt=""></a>
  <figcaption>Machine Learning Tutorial. (Kiri) Data set tanpa label . (Kanan) Clustering yang dihasilkan.  </figcaption>
</figure> 

3. Semisupervised Learning
Jika sebagian dataset memiliki label dan sebagian lagi tidak. 
Dataset : (input, correct output/ null).

<figure class="align-center">
  <a href="#"><img src="{{ '/images/semisupervised-learning.jpg' | absolute_url }}" alt=""></a>
  <figcaption>Machine Learning Tutorial. (Kiri) Sebagian data set memiliki label dan sebagian tidak . (Kanan) Klasifikasi yang dihasilkan.  </figcaption>
</figure> 

4. Reinforcement Learning
Jika dataset memiliki label tidak dalam bentuk correct output, melainkan dalam bentuk pasangan output dan ukuran seberapa bagus
output tersebut. Banyak digunakan dalam bidang robotic dan game.
Dataset : (input, output, grade for this output).
{: .text-justify}


Referensi:
1. Learning from data - Yasser Abu Mustofa
2. Deep Learning book - Yoshua Bengio
3. Pattern Recognition and Machine Learning - Bishop
4. Machine Learning Tutorial - Wei-Lun Cao