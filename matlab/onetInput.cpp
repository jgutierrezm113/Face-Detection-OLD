/*
 *	engdemo.cpp
 *
 *	A simple program to illustrate how to call MATLAB
 *	Engine functions from a C++ program.
 *
 * Copyright 1984-2011 The MathWorks, Inc.
 * All rights reserved
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "engine.h"
#define  BUFSIZE 256

// Data that is passed to matlab will be boxes (in file boxes).
// constructed as P1.x P1.y P2.x P2.y

int main() {
	Engine *ep;

         /*
         * Call engOpen with a NULL string. This starts a MATLAB process 
         * on the current host using the command "matlab".
         */
        if (!(ep = engOpen(""))) {
                fprintf(stderr, "\nCan't start MATLAB engine\n");
                return EXIT_FAILURE;
        }

        engEvalString(ep, "addpath('/usr/local/MATLAB/R2016b/toolbox/ptoolbox')");
        engEvalString(ep, "addpath('/usr/local/MATLAB/R2016b/toolbox/ptoolbox/channels')");

        // Open Image
        engEvalString(ep, "img=imread('img.jpg');");
	
	engEvalString(ep, "h=size(img,1);");
	engEvalString(ep, "w=size(img,2);");
        
        // Open File with the box info
        engEvalString(ep, "fileID = fopen('boxes.txt','r');");
        engEvalString(ep, "formatSpec = '%f %f %f %f';");
        engEvalString(ep, "sizeA = [4 Inf];");
        engEvalString(ep, "total_boxes = fscanf(fileID,formatSpec,sizeA);");
        engEvalString(ep, "total_boxes = total_boxes';");
        
        // Run algorithm
        
        engEvalString(ep, "onetInputMatlab(total_boxes,img,w,h)");
        
	engClose(ep);
	
	return EXIT_SUCCESS;
}








