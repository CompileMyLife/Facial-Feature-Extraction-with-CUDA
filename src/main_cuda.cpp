#include "image.h"
#include "haar.h"
#include "cuda_detect.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    // 1. Load the input image.
    MyImage imageObj;
    MyImage *image = &imageObj;
    if (readPgm("Face.pgm", image) == -1) {
        printf("Unable to open input image\n");
        return 1;
    }

    // 2. Compute integral images.
    MyIntImage sumObj, sqsumObj;
    MyIntImage *sum = &sumObj;
    MyIntImage *sqsum = &sqsumObj;
    createSumImage(image->width, image->height, sum);
    createSumImage(image->width, image->height, sqsum);
    integralImages(image, sum, sqsum);

    // 3. Initialize the cascade classifier.
    myCascade cascadeObj;
    myCascade *cascade = &cascadeObj;
    readTextClassifier();
    // (Set any additional cascade properties here if needed.)

    // 4. Link integral images to the cascade.
    setImageForCascadeClassifier(cascade, sum, sqsum);

    // 5. Run CUDA detection.
    int maxCandidates = 10000;  // Adjust as needed.
    float scaleFactor = 1.0f;    // Current scale factor (modify if using an image pyramid).
    runDetection(sum, sqsum, cascade, maxCandidates, scaleFactor);

    return 0;
}
