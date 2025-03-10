/*
*  PORT LINUX TO WINDOWS CHANGE LOG:
*  image.c, readPgm(), line131 //changed fopen(,'r') to fopen(,'rb') //fix sub and cntrl-z errors since windows reads 'r' as text and 'rb' as binary //This did not occur in Linux version
*  image.c, writePgm(), line207 //changed fopen(,'w') to fopen(,'wb') //fix sub and cntrl-z errors since windows writes 'w' as text and 'wb' as binary //This did not occur in Linux version
*  haar.cpp, readTextClassifier(), line670, line713 //changed fopen(,'r') to fopen(,'rb')//not completely necessary since this fopen() only reads integer numbers... good for completeness
*  haar.cpp, ScaleImage_Invoker(), line545 //changed x <= x2 ...to... x <= x2-1 //potential memory fault noted in evalWeakClassifier() function within haar.cpp ... this is a result of indexing an array not allocated in memory??? //This did not occur in Linux version
*  haar.cpp, ScaleImage_Invoker(), line546 //changed y <= y2 ...to... y <= y2-1 //potential memory fault noted in evalWeakClassifier() function within haar.cpp ... this is a result of indexing an array not allocated in memory??? //This did not occur in Linux version
*/

/* *  TU Eindhoven
 *  Eindhoven, The Netherlands
 *
 *  Name            :   faceDetection.cpp
 *
 *  Author          :   Francesco Comaschi (f.comaschi@tue.nl)
 *
 *  Date            :   November 12, 2012
 *
 *  Function        :   Main function for face detection
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

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <unistd.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>

#include "image.h"
#include "haar.h"

int main(int argc, char** argv)
{

	int flag;
    int opt;
    int rc;

    char* input_file_path = NULL;
    char* output_file_path = NULL;

    struct timespec t_start;
    struct timespec t_end;  

	int i;

    while((opt = getopt(argc, argv, "i:o:")) != -1) {
        switch(opt) {
            // Input Image
            case 'i':
                // Check that the path doesn't exist
                if (access(optarg, F_OK) != 0) {
                    fprintf(stderr, "ERROR: path to file %s does not exist\n", optarg);
                    fprintf(stderr, "Usage: %s -i [path/to/image] -o [path/to/output/image]\nExitting...\n", argv[0]);
                    exit(1);
                }

                input_file_path = optarg;
                break;

            // Output Image name
            case 'o':
                output_file_path = optarg;
                break;

            default:
                    fprintf(stderr, "Usage: %s -i [path/to/image] -o [path/to/output/image]\nExitting...\n", argv[0]);
                    exit(1);
        }
    }
    
    printf("-- entering main function --\n");


	/* detection parameters */
	float scaleFactor = 1.2;
	int minNeighbours = 1;

	printf("-- loading image --\r\n");

	MyImage imageObj;
	MyImage *image = &imageObj;               //get address of image in memory

	flag = readPgm(input_file_path, image);//read the .pgm image
	if (flag == -1)
	{
		printf( "Unable to open input image\n");
		return 1;
	}

	myCascade cascadeObj;
	myCascade *cascade = &cascadeObj;
	MySize minSize = {20, 20};
	MySize maxSize = {0, 0};

	/* classifier properties */
	cascade->n_stages=25;                   //number of strong classifier stages
	cascade->total_nodes=2913;              //number of total weak classifier notes in the cascade
	cascade->orig_window_size.height = 24;  //original window height
	cascade->orig_window_size.width = 24;   //original window width

    printf("-- loading cascade classifier --\n");
	readTextClassifier();
    printf("-- cascade classifier loaded --\n");

	std::vector<MyRect> result;

	printf("-- detecting faces --\r\n");

    // Start Timer
    rc = clock_gettime(CLOCK_REALTIME, &t_start);
    assert(rc == 0);

	result = detectObjects(image, minSize, maxSize, cascade, scaleFactor, minNeighbours);

    rc = clock_gettime(CLOCK_REALTIME, &t_end);
    assert(rc == 0);

    unsigned long long int runtime = 1000000000 * (t_end.tv_sec - t_start.tv_sec) + t_end.tv_nsec - t_start.tv_nsec; 

    printf("\nCPU detection detected %ld candidates\n", result.size());
    printf("Time = %lld nanoseconds\t(%lld.%09lld sec)\n\n", runtime, runtime / 1000000000, runtime % 1000000000);

	for(i = 0; i < (int)result.size(); i++)
	{
		MyRect r = result[i];
		drawRectangle(image, r);
		printf("Detected %d: x=%d, y=%d, width=%d, height=%d\n", 
                i, r.x, r.y, r.width, r.height);
	}

	printf("-- saving output --\r\n");
	flag = writePgm(output_file_path, image);

	printf("-- image saved --\r\n");

	/* delete image and free classifier */
	releaseTextClassifier();
	freeImage(image);

    return 0;
}
