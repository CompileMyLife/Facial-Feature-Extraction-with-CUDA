/*
 *  TU Eindhoven
 *  Eindhoven, The Netherlands
 *
 *  Name            :   haar.cpp
 *
 *  Author          :   Francesco Comaschi (f.comaschi@tue.nl)
 *
 *  Date            :   November 12, 2012
 *
 *  Function        :   Haar features evaluation for face detection
 *
 *  History         :
 *      12-11-12    :   Initial version.
 *
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program;  If not, see <http://www.gnu.org/licenses/>
 *
 * In other words, you are welcome to use, share and improve this program.
 * You are forbidden to forbid anyone else to use, share and improve
 * what you give them.   Happy coding!
 */

#include "haar.h"
#include "image.h"
#include <stdio.h>
#include "stdio-wrapper.h"

/* TODO: use matrices */
/* classifier parameters */
/************************************
 * Notes:
 * To paralleism the filter,
 * these monolithic arrays may
 * need to be splitted or duplicated
 ***********************************/
static int *stages_array;
static int *rectangles_array;
static int *weights_array;
static int *alpha1_array;
static int *alpha2_array;
static int *tree_thresh_array;
static int *stages_thresh_array;
static int **scaled_rectangles_array;


int clock_counter = 0;
float n_features = 0;


int iter_counter = 0;

/* compute integral images */
void integralImages( MyImage *src, MyIntImage *sum, MyIntImage *sqsum );

/* scale down the image */
void ScaleImage_Invoker( myCascade* _cascade, float _factor, int sum_row, int sum_col, std::vector<MyRect>& _vec);

/* compute scaled image */
void nearestNeighbor (MyImage *src, MyImage *dst);

/* debug print function */
void debugPrintIntegralImageCPU(MyIntImage *img, int numSamples);

/* rounding function */
inline  int  myRound( float value )
{
  return (int)(value + (value >= 0 ? 0.5 : -0.5));
}

/*******************************************************
 * Function: detectObjects
 * Description: It calls all the major steps
 ******************************************************/

std::vector<MyRect> detectObjects( MyImage* _img, MySize minSize, MySize maxSize, myCascade* cascade,
				   float scaleFactor, int minNeighbors)
{

  /* group overlaping windows */
  const float GROUP_EPS = 0.4f;
  /* pointer to input image */
  MyImage *img = _img;
  /***********************************
   * create structs for images
   * see haar.h for details
   * img1: normal image (unsigned char)
   * sum1: integral image (int)
   * sqsum1: square integral image (int)
   **********************************/
  MyImage image1Obj;
  MyIntImage sum1Obj;
  MyIntImage sqsum1Obj;
  /* pointers for the created structs */
  MyImage *img1 = &image1Obj;
  MyIntImage *sum1 = &sum1Obj;
  MyIntImage *sqsum1 = &sqsum1Obj;

  /********************************************************
   * allCandidates is the preliminaray face candidate,
   * which will be refined later.
   *
   * std::vector is a sequential container
   * http://en.wikipedia.org/wiki/Sequence_container_(C++)
   *
   * Each element of the std::vector is a "MyRect" struct
   * MyRect struct keeps the info of a rectangle (see haar.h)
   * The rectangle contains one face candidate
   *****************************************************/
  std::vector<MyRect> allCandidates;

  /* scaling factor */
  float factor;

  /* maxSize */
  if( maxSize.height == 0 || maxSize.width == 0 )
    {
      maxSize.height = img->height;
      maxSize.width = img->width;
    }

  /* window size of the training set */
  MySize winSize0 = cascade->orig_window_size;

  /* malloc for img1: unsigned char */
  createImage(img->width, img->height, img1);
  /* malloc for sum1: unsigned char */
  createSumImage(img->width, img->height, sum1);
  /* malloc for sqsum1: unsigned char */
  createSumImage(img->width, img->height, sqsum1);

  /* initial scaling factor */
  factor = 1;

  /* iterate over the image pyramid */
  for( factor = 1; ; factor *= scaleFactor )
    {
      /* iteration counter */
      iter_counter++;

      /* size of the image scaled up */
      MySize winSize = { myRound(winSize0.width*factor), myRound(winSize0.height*factor) };

      /* size of the image scaled down (from bigger to smaller) */
      MySize sz = { ( img->width/factor ), ( img->height/factor ) };

      /* difference between sizes of the scaled image and the original detection window */
      MySize sz1 = { sz.width - winSize0.width, sz.height - winSize0.height };

      /* if the actual scaled image is smaller than the original detection window, break */
      if( sz1.width < 0 || sz1.height < 0 )
	break;

      /* if a minSize different from the original detection window is specified, continue to the next scaling */
      if( winSize.width < minSize.width || winSize.height < minSize.height )
	continue;

      /*************************************
       * Set the width and height of
       * img1: normal image (unsigned char)
       * sum1: integral image (int)
       * sqsum1: squared integral image (int)
       * see image.c for details
       ************************************/
      setImage(sz.width, sz.height, img1);
      setSumImage(sz.width, sz.height, sum1);
      setSumImage(sz.width, sz.height, sqsum1);

      /***************************************
       * Compute-intensive step:
       * building image pyramid by downsampling
       * downsampling using nearest neighbor
       **************************************/
      nearestNeighbor(img, img1);

      /***************************************************
       * Compute-intensive step:
       * At each scale of the image pyramid,
       * compute a new integral and squared integral image
       ***************************************************/
      integralImages(img1, sum1, sqsum1);

      // For debugging: print a summary of the computed integral image.

      if (iter_counter == 2) {
      printf("DEBUG: Scale iteration %d, factor = %.3f\n", iter_counter, factor);
      debugPrintIntegralImageCPU(sum1, 10);
      }

      /* sets images for haar classifier cascade */
      /**************************************************
       * Note:
       * Summing pixels within a haar window is done by
       * using four corners of the integral image:
       * http://en.wikipedia.org/wiki/Summed_area_table
       *
       * This function loads the four corners,
       * but does not do compuation based on four coners.
       * The computation is done next in ScaleImage_Invoker
       *************************************************/
      setImageForCascadeClassifier( cascade, sum1, sqsum1);

      /* print out for each scale of the image pyramid */
      printf("detecting faces, iter := %d\n", iter_counter);

      /****************************************************
       * Process the current scale with the cascaded fitler.
       * The main computations are invoked by this function.
       * Optimization oppurtunity:
       * the same cascade filter is invoked each time
       ***************************************************/
      ScaleImage_Invoker(cascade, factor, sum1->height, sum1->width,
			 allCandidates);
    } /* end of the factor loop, finish all scales in pyramid*/

  if( minNeighbors != 0)
    {
      groupRectangles(allCandidates, minNeighbors, GROUP_EPS);
    }

  freeImage(img1);
  freeSumImage(sum1);
  freeSumImage(sqsum1);
  return allCandidates;

}

/***********************************************
 * Note:
 * The int_sqrt is softwar integer squre root.
 * GPU has hardware for floating squre root (sqrtf).
 * In GPU, it is wise to convert an int variable
 * into floating point, and use HW sqrtf function.
 * More info:
 * http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#standard-functions
 **********************************************/
/*****************************************************
 * The int_sqrt is only used in runCascadeClassifier
 * If you want to replace int_sqrt with HW sqrtf in GPU,
 * simple look into the runCascadeClassifier function.
 *****************************************************/
unsigned int int_sqrt (unsigned int value)
{
  int i;
  unsigned int a = 0, b = 0, c = 0;
  for (i=0; i < (32 >> 1); i++)
    {
      c<<= 2;
#define UPPERBITS(value) (value>>30)
      c += UPPERBITS(value);
#undef UPPERBITS
      value <<= 2;
      a <<= 1;
      b = (a<<1) | 1;
      if (c >= b)
	{
	  c -= b;
	  a++;
	}
    }
  return a;
}

void debugPrintIntegralImageCPU(MyIntImage *img, int numSamples) {
  int width = img->width;
  int height = img->height;
  int total = width * height;
  printf("Integral image summary: width = %d, height = %d, total values = %d\n", width, height, total);
  
  // Print the four corner values
  printf("Top-left (index 0): %d\n", img->data[0]);
  printf("Top-right (index %d): %d\n", width - 1, img->data[width - 1]);
  printf("Bottom-left (index %d): %d\n", (height - 1) * width, img->data[(height - 1) * width]);
  printf("Bottom-right (index %d): %d\n", total - 1, img->data[total - 1]);
  
  // Print an evenly spaced subset of values.
  int step = total / numSamples;
  if (step < 1)
      step = 1;
  printf("Printing %d sample values (every %d-th value):\n", numSamples, step);
  for (int i = 0; i < total; i += step) {
      printf("Index %d: %d\n", i, img->data[i]);
  }
}


void setImageForCascadeClassifier( myCascade* _cascade, MyIntImage* _sum, MyIntImage* _sqsum)
{
  MyIntImage *sum = _sum;
  MyIntImage *sqsum = _sqsum;
  myCascade* cascade = _cascade;
  int i, j, k;
  MyRect equRect;
  int r_index = 0;
  int w_index = 0;
  MyRect tr;

  cascade->sum = *sum;
  cascade->sqsum = *sqsum;

  equRect.x = equRect.y = 0;
  equRect.width = cascade->orig_window_size.width;
  equRect.height = cascade->orig_window_size.height;

  cascade->inv_window_area = equRect.width*equRect.height;

  cascade->p0 = (sum->data) ;
  cascade->p1 = (sum->data +  equRect.width - 1) ;
  cascade->p2 = (sum->data + sum->width*(equRect.height - 1));
  cascade->p3 = (sum->data + sum->width*(equRect.height - 1) + equRect.width - 1);
  cascade->pq0 = (sqsum->data);
  cascade->pq1 = (sqsum->data +  equRect.width - 1) ;
  cascade->pq2 = (sqsum->data + sqsum->width*(equRect.height - 1));
  cascade->pq3 = (sqsum->data + sqsum->width*(equRect.height - 1) + equRect.width - 1);

  /****************************************
   * Load the index of the four corners
   * of the filter rectangle
   **************************************/

  /* loop over the number of stages */
  for( i = 0; i < cascade->n_stages; i++ )
    {
      /* loop over the number of haar features */
      for( j = 0; j < stages_array[i]; j++ )
	{
	  int nr = 3;
	  /* loop over the number of rectangles */
	  for( k = 0; k < nr; k++ )
	    {
	      tr.x = rectangles_array[r_index + k*4];
	      tr.width = rectangles_array[r_index + 2 + k*4];
	      tr.y = rectangles_array[r_index + 1 + k*4];
	      tr.height = rectangles_array[r_index + 3 + k*4];
	      if (k < 2)
		{
		  scaled_rectangles_array[r_index + k*4] = (sum->data + sum->width*(tr.y ) + (tr.x )) ;
		  scaled_rectangles_array[r_index + k*4 + 1] = (sum->data + sum->width*(tr.y ) + (tr.x  + tr.width)) ;
		  scaled_rectangles_array[r_index + k*4 + 2] = (sum->data + sum->width*(tr.y  + tr.height) + (tr.x ));
		  scaled_rectangles_array[r_index + k*4 + 3] = (sum->data + sum->width*(tr.y  + tr.height) + (tr.x  + tr.width));
		}
	      else
		{
		  if ((tr.x == 0)&& (tr.y == 0) &&(tr.width == 0) &&(tr.height == 0))
		    {
		      scaled_rectangles_array[r_index + k*4] = NULL ;
		      scaled_rectangles_array[r_index + k*4 + 1] = NULL ;
		      scaled_rectangles_array[r_index + k*4 + 2] = NULL;
		      scaled_rectangles_array[r_index + k*4 + 3] = NULL;
		    }
		  else
		    {
		      scaled_rectangles_array[r_index + k*4] = (sum->data + sum->width*(tr.y ) + (tr.x )) ;
		      scaled_rectangles_array[r_index + k*4 + 1] = (sum->data + sum->width*(tr.y ) + (tr.x  + tr.width)) ;
		      scaled_rectangles_array[r_index + k*4 + 2] = (sum->data + sum->width*(tr.y  + tr.height) + (tr.x ));
		      scaled_rectangles_array[r_index + k*4 + 3] = (sum->data + sum->width*(tr.y  + tr.height) + (tr.x  + tr.width));
		    }
		} /* end of branch if(k<2) */
	    } /* end of k loop*/
	  r_index+=12;
	  w_index+=3;
	} /* end of j loop */
    } /* end i loop */
}


/****************************************************
 * evalWeakClassifier:
 * the actual computation of a haar filter.
 * More info:
 * http://en.wikipedia.org/wiki/Haar-like_features
 ***************************************************/
inline float evalWeakClassifier(myCascade* cascade, int variance_norm_factor, int p_offset, int tree_index, int w_index, int r_index )
{

  /* the node threshold is multiplied by the standard deviation of the image */
  int t = tree_thresh_array[tree_index] * variance_norm_factor;

  int sum1 = (*(scaled_rectangles_array[r_index] + p_offset)
	     - *(scaled_rectangles_array[r_index + 1] + p_offset)
	     - *(scaled_rectangles_array[r_index + 2] + p_offset)
	     + *(scaled_rectangles_array[r_index + 3] + p_offset))
    * weights_array[w_index];

//printf("sum1: %d\n",sum);

  int sum2 = (*(scaled_rectangles_array[r_index+4] + p_offset)
	  - *(scaled_rectangles_array[r_index + 5] + p_offset)
	  - *(scaled_rectangles_array[r_index + 6] + p_offset)
	  + *(scaled_rectangles_array[r_index + 7] + p_offset))
    * weights_array[w_index + 1];

//printf("sum2: %d\n",sum);

  int sum3 = 0;
  if ((scaled_rectangles_array[r_index+8] != NULL))
    sum3 = (*(scaled_rectangles_array[r_index+8] + p_offset)
	    - *(scaled_rectangles_array[r_index + 9] + p_offset)
	    - *(scaled_rectangles_array[r_index + 10] + p_offset)
	    + *(scaled_rectangles_array[r_index + 11] + p_offset))
      * weights_array[w_index + 2];

//printf("sum3: %d\n",sum);

  int final_sum = sum1 + sum2 + sum3;

  // Debug block for candidate (2015,863) at Stage 0, Feature 0:
  // p_offset was computed in runCascadeClassifier as: 
  //    p_offset = pt.y * cascade->sum.width + pt.x.
  // So candidate (2015,863) should have p_offset == 2015 + 863 * cascade->sum.width.
  if (p_offset == 2015 + (863 * cascade->sum.width) && tree_index == 0) {
    int candidate_x = 2015;
    int candidate_y = 863;
    printf("[CPU DEBUG] Candidate (%d,%d):\n", candidate_x, candidate_y);

    // For the first rectangle: the classifier data (rectangles_array) holds the offsets and sizes.
    int rx = rectangles_array[r_index];      // rectangle offset x
    int ry = rectangles_array[r_index + 1];    // rectangle offset y
    int rw = rectangles_array[r_index + 2];    // rectangle width
    int rh = rectangles_array[r_index + 3];    // rectangle height

    // Compute the actual rectangle coordinates by adding the candidate base coordinates.
    int tl_x = candidate_x + rx;
    int tl_y = candidate_y + ry;
    int br_x = tl_x + rw;
    int br_y = tl_y + rh;
    printf("[CPU DEBUG] First rectangle: tl=(%d,%d), br=(%d,%d)\n", tl_x, tl_y, br_x, br_y);

    // Compute the indices into the integral image.
    int width = cascade->sum.width;
    int idx_tl = tl_y * width + tl_x;
    int idx_br = br_y * width + br_x;
    int val_tl = cascade->sum.data[idx_tl];
    int val_br = cascade->sum.data[idx_br];
    printf("[CPU DEBUG] Integral image indices: tl = %d, br = %d\n", idx_tl, idx_br);
    printf("[CPU DEBUG] p0[tl] = %d, p0[br] = %d\n", val_tl, val_br);

    printf("[CPU DEBUG] Candidate (%d,%d), Stage 0, Feature 0: sum1 = %d, sum2 = %d, sum3 = %d, final_sum = %d\n",
           candidate_x, candidate_y, sum1, sum2, sum3, final_sum);
}

  if(final_sum >= t){
    
               return (float)alpha2_array[tree_index];
  } else {
    
      return (float)alpha1_array[tree_index];
  }

}



int runCascadeClassifier( myCascade* _cascade, MyPoint pt, int start_stage )
{

  int p_offset, pq_offset;
  int i, j;
  unsigned int mean;
  unsigned int variance_norm_factor;
  int haar_counter = 0;
  int w_index = 0;
  int r_index = 0;
  int stage_sum;
  myCascade* cascade;
  cascade = _cascade;

  p_offset = pt.y * (cascade->sum.width) + pt.x;
  pq_offset = pt.y * (cascade->sqsum.width) + pt.x;

  // Debug: Print candidate offsets and integral image dimensions.
  /*printf("[CPU DEBUG] Candidate=(%d,%d): p_offset=%d, pq_offset=%d, sum dims=(%d,%d), sqsum dims=(%d,%d)\n",
    pt.x, pt.y, p_offset, pq_offset,
    cascade->sum.width, cascade->sum.height,
    cascade->sqsum.width, cascade->sqsum.height);*/

  /**************************************************************************
   * Image normalization
   * mean is the mean of the pixels in the detection window
   * cascade->pqi[pq_offset] are the squared pixel values (using the squared integral image)
   * inv_window_area is 1 over the total number of pixels in the detection window
   *************************************************************************/

  variance_norm_factor =  (cascade->pq0[pq_offset] - cascade->pq1[pq_offset] - cascade->pq2[pq_offset] + cascade->pq3[pq_offset]);
  mean = (cascade->p0[p_offset] - cascade->p1[p_offset] - cascade->p2[p_offset] + cascade->p3[p_offset]);

  //printf("[CPU DEBUG] Candidate=(%d,%d): mean=%u, initial var=%u\n", pt.x, pt.y, mean, variance_norm_factor);

  variance_norm_factor = (variance_norm_factor*cascade->inv_window_area);
  variance_norm_factor =  variance_norm_factor - mean*mean;

  /***********************************************
   * Note:
   * The int_sqrt is softwar integer squre root.
   * GPU has hardware for floating squre root (sqrtf).
   * In GPU, it is wise to convert the variance norm
   * into floating point, and use HW sqrtf function.
   * More info:
   * http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#standard-functions
   **********************************************/
  if( variance_norm_factor > 0 )
    variance_norm_factor = int_sqrt(variance_norm_factor);
  else
    variance_norm_factor = 1;

  //printf("[CPU DEBUG] Candidate=(%d,%d): final variance_norm=%u\n", pt.x, pt.y, variance_norm_factor);

  /**************************************************
   * The major computation happens here.
   * For each scale in the image pyramid,
   * and for each shifted step of the filter,
   * send the shifted window through cascade filter.
   *
   * Note:
   *
   * Stages in the cascade filter are independent.
   * However, a face can be rejected by any stage.
   * Running stages in parallel delays the rejection,
   * which induces unnecessary computation.
   *
   * Filters in the same stage are also independent,
   * except that filter results need to be merged,
   * and compared with a per-stage threshold.
   *************************************************/
  for( i = start_stage; i < cascade->n_stages; i++ )
    {

      /****************************************************
       * A shared variable that induces false dependency
       *
       * To avoid it from limiting parallelism,
       * we can duplicate it multiple times,
       * e.g., using stage_sum_array[number_of_threads].
       * Then threads only need to sync at the end
       ***************************************************/
      stage_sum = 0;
      int num_features = stages_array[i];
      //printf("[CPU DEBUG] Stage %d: Candidate=(%d,%d), num_features=%d\n", i, pt.x, pt.y, num_features);

      for( j = 0; j < stages_array[i]; j++ )
	    {
        /**************************************************
         * Send the shifted window to a haar filter.
         **************************************************/
          
        int feature_result = evalWeakClassifier(cascade, variance_norm_factor, p_offset, haar_counter, w_index, r_index);
        //printf("[CPU DEBUG] Candidate=(%d,%d), Stage %d, Feature %d: result=%d\n", pt.x, pt.y, i, j, feature_result);
        stage_sum += feature_result;
        n_features++;
        haar_counter++;
        w_index+=3;
        r_index+=12;
	    } /* end of j loop */
      
     // printf("[CPU DEBUG] Stage %d: Candidate=(%d,%d), stage_sum=%d, threshold=%d\n", i, pt.x, pt.y, stage_sum, stages_thresh_array[i]);
      
      /**************************************************************
       * threshold of the stage.
       * If the sum is below the threshold,
       * no faces are detected,
       * and the search is abandoned at the i-th stage (-i).
       * Otherwise, a face is detected (1)
       **************************************************************/

      /* the number "0.4" is empirically chosen for 5kk73 */
      if( stage_sum < 0.4*stages_thresh_array[i] ){
	return -i;
      } /* end of the per-stage thresholding */
    } /* end of i loop */
  return 1;
}


void ScaleImage_Invoker( myCascade* _cascade, float _factor, int sum_row, int sum_col, std::vector<MyRect>& _vec)
{

  myCascade* cascade = _cascade;

  float factor = _factor;
  MyPoint p;
  int result;
  int y1, y2, x2, x, y, step;
  std::vector<MyRect> *vec = &_vec;

  MySize winSize0 = cascade->orig_window_size;
  MySize winSize;

  winSize.width =  myRound(winSize0.width*factor);
  winSize.height =  myRound(winSize0.height*factor);
  y1 = 0;

  /********************************************
  * When filter window shifts to image boarder,
  * some margin need to be kept
  *********************************************/
  y2 = sum_row - winSize0.height;
  x2 = sum_col - winSize0.width;

  /********************************************
   * Step size of filter window shifting
   * Reducing step makes program faster,
   * but decreases quality of detection.
   * example:
   * step = factor > 2 ? 1 : 2;
   *
   * For 5kk73,
   * the factor and step can be kept constant,
   * unless you want to change input image.
   *
   * The step size is set to 1 for 5kk73,
   * i.e., shift the filter window by 1 pixel.
   *******************************************/
  step = 1;

  /**********************************************
   * Shift the filter window over the image.
   * Each shift step is independent.
   * Shared data structure may limit parallelism.
   *
   * Some random hints (may or may not work):
   * Split or duplicate data structure.
   * Merge functions/loops to increase locality
   * Tiling to increase computation-to-memory ratio
   *********************************************/
  for( x = 0; x <= x2-1; x += step )        //changed x <= x2 ...to... x <= x2-1
    for( y = y1; y <= y2-1; y += step )     //changed y <= y2 ...to... y <= y2-1
      {
	p.x = x;
	p.y = y;

	/*********************************************
	 * Optimization Oppotunity:
	 * The same cascade filter is used each time
	 ********************************************/
	result = runCascadeClassifier( cascade, p, 0 );

	/*******************************************************
	 * If a face is detected,
	 * record the coordinates of the filter window
	 * the "push_back" function is from std:vec, more info:
	 * http://en.wikipedia.org/wiki/Sequence_container_(C++)
	 *
	 * Note that, if the filter runs on GPUs,
	 * the push_back operation is not possible on GPUs.
	 * The GPU may need to use a simpler data structure,
	 * e.g., an array, to store the coordinates of face,
	 * which can be later memcpy from GPU to CPU to do push_back
	 *******************************************************/
	if( result > 0 )
	  {
	    MyRect r = {myRound(x*factor), myRound(y*factor), winSize.width, winSize.height};
	    vec->push_back(r);
	  }
      }
}

/*****************************************************
 * Compute the integral image (and squared integral)
 * Integral image helps quickly sum up an area.
 * More info:
 * http://en.wikipedia.org/wiki/Summed_area_table
 ****************************************************/
void integralImages( MyImage *src, MyIntImage *sum, MyIntImage *sqsum )
{
  int x, y, s, sq, t, tq;
  unsigned char it;
  int height = src->height;
  int width = src->width;
  unsigned char *data = src->data;
  int * sumData = sum->data;
  int * sqsumData = sqsum->data;
  for( y = 0; y < height; y++)
    {
      s = 0;
      sq = 0;
      /* loop over the number of columns */
      for( x = 0; x < width; x ++)
	{
	  it = data[y*width+x];
	  /* sum of the current row (integer)*/
	  s += it;
	  sq += it*it;

	  t = s;
	  tq = sq;
	  if (y != 0)
	    {
	      t += sumData[(y-1)*width+x];
	      tq += sqsumData[(y-1)*width+x];
	    }
	  sumData[y*width+x]=t;
	  sqsumData[y*width+x]=tq;
	}
    }
}

/***********************************************************
 * This function downsample an image using nearest neighbor
 * It is used to build the image pyramid
 **********************************************************/
void nearestNeighbor (MyImage *src, MyImage *dst)
{

  int y;
  int j;
  int x;
  int i;
  unsigned char* t;
  unsigned char* p;
  int w1 = src->width;
  int h1 = src->height;
  int w2 = dst->width;
  int h2 = dst->height;

  int rat = 0;

  unsigned char* src_data = src->data;
  unsigned char* dst_data = dst->data;


  int x_ratio = (int)((w1<<16)/w2) +1;
  int y_ratio = (int)((h1<<16)/h2) +1;

  for (i=0;i<h2;i++)
    {
      t = dst_data + i*w2;
      y = ((i*y_ratio)>>16);
      p = src_data + y*w1;
      rat = 0;
      for (j=0;j<w2;j++)
	{
	  x = (rat>>16);
	  *t++ = p[x];
	  rat += x_ratio;
	}
    }
}

void readTextClassifier()//(myCascade * cascade)
{
  /*number of stages of the cascade classifier*/
  int stages;
  /*total number of weak classifiers (one node each)*/
  int total_nodes = 0;
  int i, j, k, l;
  char mystring [12];
  int r_index = 0;
  int w_index = 0;
  int tree_index = 0;
  FILE *finfo = fopen("info.txt", "rb");

  /**************************************************
  * how many stages are in the cascaded filter?
  * the first line of info.txt is the number of stages
  * (in the 5kk73 example, there are 25 stages)
  **************************************************/
  if ( fgets (mystring , 12 , finfo) != NULL )
    {
      stages = atoi(mystring);
    }
  i = 0;

  stages_array = (int *)malloc(sizeof(int)*stages);


  // -- read how many weak classifiers per stage (subsequent lines in info.txt) --
  while (fgets(mystring, 12, finfo) != NULL)
  {
      stages_array[i] = atoi(mystring);
      total_nodes += stages_array[i];
      i++;
  }
  fclose(finfo);

  /**************************************************
   * how many filters in each stage?
   * They are specified in info.txt,
   * starting from second line.
   * (in the 5kk73 example, from line 2 to line 26)
   *************************************************/
  while ( fgets (mystring , 12 , finfo) != NULL )
    {
      stages_array[i] = atoi(mystring);
      total_nodes += stages_array[i];
      i++;
    }
  fclose(finfo);


  /* TODO: use matrices where appropriate */
  /***********************************************
   * Allocate a lot of array structures
   * Note that, to increase parallelism,
   * some arrays need to be splitted or duplicated
   **********************************************/
  rectangles_array = (int *)malloc(sizeof(int)*total_nodes*12);
  scaled_rectangles_array = (int **)malloc(sizeof(int*)*total_nodes*12);
  weights_array = (int *)malloc(sizeof(int)*total_nodes*3);
  alpha1_array = (int*)malloc(sizeof(int)*total_nodes);
  alpha2_array = (int*)malloc(sizeof(int)*total_nodes);
  tree_thresh_array = (int*)malloc(sizeof(int)*total_nodes);
  stages_thresh_array = (int*)malloc(sizeof(int)*stages);
  FILE *fp = fopen("class.txt", "rb");

  /******************************************
   * Read the filter parameters in class.txt
   *
   * Each stage of the cascaded filter has:
   * 18 parameter per filter x tilter per stage
   * + 1 threshold per stage
   *
   * For example, in 5kk73,
   * the first stage has 9 filters,
   * the first stage is specified using
   * 18 * 9 + 1 = 163 parameters
   * They are line 1 to 163 of class.txt
   *
   * The 18 parameters for each filter are:
   * 1 to 4: coordinates of rectangle 1
   * 5: weight of rectangle 1
   * 6 to 9: coordinates of rectangle 2
   * 10: weight of rectangle 2
   * 11 to 14: coordinates of rectangle 3
   * 15: weight of rectangle 3
   * 16: threshold of the filter
   * 17: alpha 1 of the filter
   * 18: alpha 2 of the filter
   ******************************************/

  /* loop over n of stages */
  for (i = 0; i < stages; i++)
    {    /* loop over n of trees */
      for (j = 0; j < stages_array[i]; j++)
	{	/* loop over n of rectangular features */
	  for(k = 0; k < 3; k++)
	    {	/* loop over the n of vertices */
	      for (l = 0; l <4; l++)
		{
		  if (fgets (mystring , 12 , fp) != NULL)
		    rectangles_array[r_index] = atoi(mystring);
		  else
		    break;
		  r_index++;
		} /* end of l loop */
	      if (fgets (mystring , 12 , fp) != NULL)
		{
		  weights_array[w_index] = atoi(mystring);
		  /* Shift value to avoid overflow in the haar evaluation */
		  /*TODO: make more general */
		  /*weights_array[w_index]>>=8; */
		}
	      else
		break;
	      w_index++;
	    } /* end of k loop */
	  if (fgets (mystring , 12 , fp) != NULL)
	    tree_thresh_array[tree_index]= atoi(mystring);
	  else
	    break;
	  if (fgets (mystring , 12 , fp) != NULL)
	    alpha1_array[tree_index]= atoi(mystring);
	  else
	    break;
	  if (fgets (mystring , 12 , fp) != NULL)
	    alpha2_array[tree_index]= atoi(mystring);
	  else
	    break;
	  tree_index++;
	  if (j == stages_array[i]-1)
	    {
	      if (fgets (mystring , 12 , fp) != NULL)
		stages_thresh_array[i] = atoi(mystring);
	      else
		break;
	    }
	} /* end of j loop */
    } /* end of i loop */
  fclose(fp);
  
  /*
  /* Debug prints: immediately after array population 
  printf("\n--- Debug: stages_array (first 10 elements) ---\n");
  for (int idx = 0; idx < ((stages < 10) ? stages : 10); idx++)
  {
      printf("  stages_array[%d] = %d\n", idx, stages_array[idx]);
  }

  printf("\n--- Debug: stages_thresh_array (first 10 elements) ---\n");
  for (int idx = 0; idx < ((stages < 10) ? stages : 10); idx++)
  {
      printf("  stages_thresh_array[%d] = %d\n", idx, stages_thresh_array[idx]);
  }

  printf("\n--- Debug: rectangles_array (first 10 elements) ---\n");
  int totalRectElems = total_nodes * 12;
  for (int idx = 0; idx < ((totalRectElems < 10) ? totalRectElems : 10); idx++)
  {
      printf("  rectangles_array[%d] = %d\n", idx, rectangles_array[idx]);
  }

  printf("\n--- Debug: weights_array (first 10 elements) ---\n");
  int totalWeightElems = total_nodes * 3;
  for (int idx = 0; idx < ((totalWeightElems < 10) ? totalWeightElems : 10); idx++)
  {
      printf("  weights_array[%d] = %d\n", idx, weights_array[idx]);
  }
      */
}


void releaseTextClassifier()
{
  free(stages_array);
  free(rectangles_array);
  free(scaled_rectangles_array);
  free(weights_array);
  free(tree_thresh_array);
  free(alpha1_array);
  free(alpha2_array);
  free(stages_thresh_array);
}
/* End of file. */
