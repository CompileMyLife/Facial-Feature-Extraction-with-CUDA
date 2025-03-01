/*
 *  TU Eindhoven
 *  Eindhoven, The Netherlands
 *
 *  Name            :   haar.h
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
 * along with this program; If not, see <http://www.gnu.org/licenses/>
 *
 * In other words, you are welcome to use, share and improve this program.
 * You are forbidden to forbid anyone else to use, share and improve
 * what you give them.   Happy coding!
 */

#ifndef __HAAR_H__
#define __HAAR_H__

#include <stdio.h>
#include <stdlib.h>
#include "image.h"
#include "stdio-wrapper.h"

#ifdef __cplusplus
#include <vector>
#endif

#define MAXLABELS 50

#ifdef __cplusplus
extern "C" {
#endif

    /* C-Compatible Type Definitions */

    typedef int sumtype;
    typedef int sqsumtype;

    typedef struct {
        int x;
        int y;
    } MyPoint;

    typedef struct {
        int width;
        int height;
    } MySize;

    typedef struct {
        int x;
        int y;
        int width;
        int height;
    } MyRect;

    typedef struct {
        int n_stages;
        int total_nodes;
        float scale;
        MySize orig_window_size;
        int inv_window_area;
        MyIntImage sum;
        MyIntImage sqsum;
        sqsumtype* pq0, * pq1, * pq2, * pq3;
        sumtype* p0, * p1, * p2, * p3;
    } myCascade;

    /* C-Compatible Function Declarations */

    /* Sets images for Haar classifier cascade */
    void setImageForCascadeClassifier(myCascade* cascade, MyIntImage* sum, MyIntImage* sqsum);

    /* Runs the cascade on the specified window */
    int runCascadeClassifier(myCascade* cascade, MyPoint pt, int start_stage);

    /* Reads the classifier file into memory */
    void readTextClassifier();

    /* Releases classifier resources */
    void releaseTextClassifier();

    /* Draws white bounding boxes around detected faces */
    void drawRectangle(MyImage* image, MyRect r);

    /* Computes integral images (and squared integral images) from a source image */
    void integralImages(MyImage* src, MyIntImage* sum, MyIntImage* sqsum);

#ifdef __cplusplus
} // End of extern "C"
#endif

#ifdef __cplusplus
/* C++-Only Function Declarations (using std::vector) */

std::vector<MyRect> detectObjects(MyImage* image, MySize minSize, MySize maxSize,
    myCascade* cascade, float scale_factor, int min_neighbors);

void groupRectangles(std::vector<MyRect>& _vec, int groupThreshold, float eps);
#endif

#endif // __HAAR_H__
