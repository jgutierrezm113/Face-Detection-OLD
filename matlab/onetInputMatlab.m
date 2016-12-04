function onetInputMatlab(total_boxes,img,w,h)
        numbox=size(total_boxes,1);
        total_boxes=fix(total_boxes);
        [dy edy dx edx y ey x ex tmpw tmph]=pad(total_boxes,w,h);
        tempimg=zeros(48,48,3,numbox);
        for k=1:numbox
                tmp=zeros(tmph(k),tmpw(k),3);
                tmp(dy(k):edy(k),dx(k):edx(k),:)=img(y(k):ey(k),x(k):ex(k),:);
                tempimg(:,:,:,k)=imResample(tmp,[48 48],'bilinear');
        end
        tempimg=(tempimg-127.5)*0.0078125;

        %Write data into file
        fid = fopen('imgout.txt', 'wt'); % Open for writing
        for i=1:size(tempimg,4)
                for j=1:size(tempimg,3)
                        for y=1:size(tempimg,2)
                                for x=1:size(tempimg,1)
                                        fprintf(fid, '%f\n', tempimg(x,y,j,i));
                                end
                        end
                end
        end
        fclose(fid);
end

