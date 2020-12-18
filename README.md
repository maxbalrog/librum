# librum - Pipeline for booktitle detection

## **Problem statement**

Managing and finding books in large collections often leads to tedious manual work. In addition, sometimes books may be missing or misplaced.
We can build a following algorithm:
* Given a bookshelf photo, it segments different book spines.
* On each book spine it locates the title (and splits in into separate characters).
* Text recognition algorithm transfers title photo into actual text which could be used to build a book database (alternatively, to find a certain book).


## **Used pipeline**

Good examples of text localization:
![](images.00.png)

Tesseract OCR was used with Russian language flag.
For training a cyrillic letters classifier we used dataset: [CoMNIST](https://github.com/GregVial/CoMNIST).

## **Literature**
1.  [Smart Library: Identifying Books in a Library using Richly Supervised Deep
Scene Text Reading](https://arxiv.org/pdf/1611.07385.pdf). Use text/non-text CNN to locate book spines, segment separate words and then predict words with CNN-RNN architecture.
2. [A Framework for Recognition Books on Bookshelves](https://www.researchgate.net/publication/220778125_A_Framework_for_Recognition_Books_on_Bookshelves). Use Canny edge map, Hough lines and calculate dominant vanishing point. Words are clustered in Canny edge map through dilation operator. 
3. [Viewpoint-Independent Book Spine Segmentation](https://www.researchgate.net/publication/269299980_Viewpoint-independent_book_spine_segmentation).
4. [Book spine segmentation for various book orientations](https://www.researchgate.net/publication/300412373_Book_spine_segmentation_for_various_book_orientations). Book spine edge map is segmented through morphological reconstruction and L0-gradient minimization. After that, SVM classifier recovers missing boundaries. 
5. Some ideas were taken from this [blog](https://www.cs.bgu.ac.il/~ben-shahar/Teaching/Computational-Vision/StudentProjects/ICBV151/ICBV-2015-1-PavelRubinson/index.php).
