InputImg = zeros(500,500,3);
for i = 1:400
    if i>=251
        InputImg(i,i,1) = 1;
        InputImg(i,401-i,1) = 1;
    end
end
imshow(InputImg);

[EdgeImg, threshold]= edge(rgb2gray(InputImg),'canny');