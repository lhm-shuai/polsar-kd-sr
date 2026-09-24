function [ T11,T12,T13,T22,T23,T33 ] = C3_T3( C11,C12,C13,C22,C23,C33 )
% 将3x3平均极化协方差矩阵转化为3x3极化相干矩阵
% [ T11,T12,T13,T22,T23,T33 ] = C3_T3( C11,C12,C13,C22,C23,C33 )

% 输入参数: C11,C12,C13,C22,C23,C33--输入3*3平均极化协方差矩阵的元素(注意为M*N的矩阵!)

% 输出参数: T11,T12,T13,T22,T23,T33--输入3*3平均极化相干矩阵的元素(注意为M*N的矩阵!)

% Author  : Sinong Quan
% Creation: 2016.11.04
% Updata: 2016.11.04

C12_re = real(C12);
C12_im = imag(C12);

C13_re = real(C13);
C13_im = imag(C13);

C23_re = real(C23);
C23_im = imag(C23);

j = sqrt(-1);

T11 = 0.5 * (C11 + C33  +2 * C13_re);

T12_re = (C11 - C33) / 2;
T12_im =  -C13_im;
T12 = T12_re + j * T12_im;


T13_re = (C12_re + C23_re) / sqrt(2);
T13_im = (C12_im - C23_im) / sqrt(2);
T13 = T13_re + j * T13_im;

T22 = 0.5 * (C11 + C33 - 2 * C13_re);

T23_re = (C12_re - C23_re) / sqrt(2);
T23_im = (C12_im + C23_im) / sqrt(2);
T23 = T23_re + j * T23_im;

T33 = C22;




% M_in[T311][lig][col] = (C11 + 2 * C13_re + C33) / 2;
% M_in[T312_re][lig][col] = (C11 - C33) / 2;
% M_in[T312_im][lig][col] = -C13_im;
% M_in[T313_re][lig][col] = (C12_re + C23_re) / sqrt(2);
% M_in[T313_im][lig][col] = (C12_im - C23_im) / sqrt(2);
% M_in[T322][lig][col] = (C11 - 2 * C13_re + C33) / 2;
% M_in[T323_re][lig][col] = (C12_re - C23_re) / sqrt(2);
% M_in[T323_im][lig][col] = (C12_im + C23_im) / sqrt(2);
% M_in[T333][lig][col] = C22;
end

