function [ Podd,Pdbl,Pvol,Phel ] = yamaguchi_4components_T3( T11,T12,T13,T22,T23,T33,mode )
% 对3×3的平均相干矩阵进行Yamaguchi四成分分解
% [Podd,Pdbl,Pvol，Phel] = yamaguchi_4components_T3( T11,T12,T13,T22,T23,T33 )
% 输入参数: T11,T12,T13,T22,T23,T33--输入3*3平均相干矩阵的元素(注意为M*N的矩阵!),由矢量[Shh sqrt(2)*Shv Svv].'的外积得到
%          mode--判断是否需要进行极化方位角补偿或者相位角补偿（0,1,2）
%          mode==0:表示进行极化方位角补偿处理（使得T23的实部为零），但是只进行yamaguchi四成分分解（Y4R）
%          mode==1:表示进行极化方位角补偿处理（使得T23的实部为零），既进行yamaguchi四成分分解，又考虑二面角体散射模型（S4R）
%          mode==2:表示进行极化方位角补偿以及相位角处理（使得T23为零），既进行yamaguchi四成分分解，又考虑二面角体散射模型（G4U）

% 输出参数: Podd--奇次散射分量功率(注意为M*N的矩阵!)
%          Pdbl--偶次散射分量功率(注意为M*N的矩阵!)
%          Pvol--体散射分量功率(注意为M*N的矩阵!)
%          Phel--螺旋体分量功率（注意为M*N的矩阵!）

% Method: Y. Yamaguchi, T. Moriyama, M. Ishido, and H. Yamada, "Four-component
%         scattering model for polarimetric SAR image decomposition," IEEE Trans.
%         Geosci. Remote Sens., vol. 43, no. 8, pp. 1699-1706, Aug. 2005.
%         (reference [27] of the accompanying paper)

eps = 1e-15;

[M,N] = size(T11);

% /********************************************/
% 无论如何变换，螺旋体散射能量不变
PPPhel = 2 * abs(imag(T23));
% /********************************************/

% 首先判断是否进行补偿
if ((mode == 0)||(mode == 1)) % 只进行极化方位角补偿
    % 计算极化方位角
    teta = 0.5 .* atan(2 .* real(T23) ./ (T22 - T33+ eps) );
    % 极化方位角补偿
    [ DT11,DT12_re,DT12_im,DT13_re,DT13_im,DT22,DT23_re,DT23_im,DT33 ] = unitary_rotation( T11,T12,T13,T22,T23,T33,teta );
    T11 = DT11;
    T12_re = DT12_re;
    T12_im = DT12_im;
    T13_re = DT13_re;
    T13_im = DT13_im;
    T22 = DT22;
    T23_re = DT23_re;
    T23_im = DT23_im;
    T33 = DT33;
    
    % 既进行极化方位角补偿又进行相位角补偿
elseif (mode == 2)
    % 计算极化方位角
    teta = 0.5 .* atan(2 .* real(T23) ./ (T22 - T33+ eps) );
    % 先进行极化方位角补偿
    [ DT11,DT12_re,DT12_im,DT13_re,DT13_im,DT22,DT23_re,DT23_im,DT33 ] = unitary_rotation( T11,T12,T13,T22,T23,T33,teta );
    j = sqrt(-1);
    DT12 = DT12_re + j * DT12_im;
    DT13 = DT13_re + j * DT13_im;
    DT23 = DT23_re + j * DT23_im;
    % 计算相位角
    phi = 0.5 .* atan(2.* imag(DT23) ./(DT22 - DT33+ eps) );
    % 然后进行相位角补偿
    [ DDT11,DDT12_re,DDT12_im,DDT13_re,DDT13_im,DDT22,DDT23_re,DDT23_im,DDT33 ] = unitary_transformation( DT11,DT12,DT13,DT22,DT23,DT33,phi );
    
    T11 = DDT11;
    T12_re = DDT12_re;
    T12_im = DDT12_im;
    T13_re = DDT13_re;
    T13_im = DDT13_im;
    T22 = DDT22;
    T23_re = DDT23_re;
    T23_im = DDT23_im;
    T33 = DDT33;
    
else % 不进行任何补偿
    
    T11 = T11;
    T12_re = real(T12);
    T12_im = imag(T12);
    T13_re = real(T13);
    T13_im = imag(T13);
    T22 = T22;
    T23_re = real(T23);
    T23_im = imag(T23);
    T33 = T33;
  
end

% 矩阵矢量化
T11 = reshape(T11,1,M*N);
T12_re = reshape(T12_re,1,M*N);
T12_im = reshape(T12_im,1,M*N);
T13_re = reshape(T13_re,1,M*N);
T13_im = reshape(T13_im,1,M*N);
T22 = reshape(T22,1,M*N);
T23_re = reshape(T23_re,1,M*N);
T23_im = reshape(T23_im,1,M*N);
T33 = reshape(T33,1,M*N);

PPPhel  = reshape(PPPhel,1,M*N);

% Yamaguchi分解
Podd = zeros(1,M*N);
Pdbl = zeros(1,M*N);
Pvol = zeros(1,M*N);
Phel = zeros(1,M*N);

for k = 1:M*N,  
    % 赋初值
    
    C1 = 0; % 区分体散射中偶极子与二次散射散射体的判决条件
    PPvol = 0; % 体散射贡献
    ratio1 = 0;% 体散射矩阵选取的判决条件
    ratio2 = 0;% 在体散射贡献小于零的情况下，体散射矩阵选取的判决条件
    HV_type = 1;% 散射主导机制
    
    % 散射功率
    fs = 0;fd = 0;fv = 0;
    
    % 参数角
    ALP_re = 0;ALP_im = 0;
    BET_re = 0;BET_im = 0;
    
    TT11 = T11(k); TT22 = T22(k); TT33 = T33(k); TT12_re = T12_re(k); TT12_im = T12_im(k);
    TT13_re = T13_re(k); TT13_im = T13_im(k);TT23_re = T23_re(k); TT23_im = T23_im(k);
    
    % 螺旋体分量贡献
    PPhel = PPPhel(k);
    
    % 功率最小值和最大值
    SpanMin = eps;
    SpanMax = TT11 + TT22 + TT33;
    
    % ---------------------------- Singh的文章的思路------------------------ %
    % 若是进行了补偿，则进行如下处理
    if ((mode ==1)||(mode ==2))
        % 判决条件（Branch Condition,用以区分体散射中偶极子与二次散射散射体！！！！！！！！！！）
        % C1 = 2Re{ fs  * beta + fd  * conj(alpha)} = 2Re{<Shh*conj(Svv)>}
        % + 7* Pv /15 + Pc/2 = T11(sita) - T22(sita) + (7/8)*T33(sita) + Pc/16;
        C1 = TT11 - TT22 + (7/8)*TT33 + PPhel/16;
        if (C1 > 0)
            % 体散射中偶极子散射占主导
            HV_type = 1;
        else
            % 体散射中树干与地面机制散射占主导（二次散射占主导）
            HV_type = 2;
        end
    % 若没有进行补偿，则默认是体散射中偶极子散射占主导 
%     else
%         HV_type 
    end
    % ---------------------------- Singh的文章的思路------------------------ %
    
    
    % 表面散射占主导
    if (HV_type == 1)
        ratio1 = 10*log10((TT11 + TT22 - 2*TT12_re)/(TT11 + TT22 + 2*TT12_re+ eps));
        if ((ratio1 > -2)&&(ratio1 <= 2))
            PPvol = 2*(2*TT33 - PPhel);
        else
            PPvol = (15/8)*(2*TT33 - PPhel);
        end
    % 二次散射占主导
    else
        PPvol = (15/16)*(2*TT33 - PPhel);
    end
    % 即SpanMax
    TP = TT11 + TT22 + TT33;
        
   % 如果 体散射贡献小于零，则退化成三成分分解（去掉螺旋体散射！！！！！（不是去掉体散射！！！！））
   if (PPvol < 0)
       % /********************************************/
       % /**** Freeman - Yamaguchi三成分分解算法   ****/
       % /********************************************/
       HHHH = (TT11 + 2 * TT12_re + TT22) / 2;
       HHVV_re = (TT11 - TT22) / 2;
       HHVV_im = -TT12_im;
       HVHV = TT33 / 2;
       VVVV = (TT11 - 2 * TT12_re + TT22) / 2;
       
       % 重新计算体散射贡献
       ratio2 = 10 * log10 (VVVV/(HHHH+ eps));
       if (ratio2 <= -2)
           fv = 15 * (HVHV / 4);
           HHHH = HHHH - 8*(fv/15);
           VVVV = VVVV - 3*(fv/15);
           HHVV_re = HHVV_re - 2*(fv/15);
       elseif (ratio2 > 2)
           fv = 15 * (HVHV/4);
           HHHH = HHHH - 3*(fv/15);
           VVVV = VVVV - 8*(fv/15);
           HHVV_re = HHVV_re - 2*(fv/15);
       else
           fv = 8 * (HVHV/2);
           HHHH = HHHH - 3*(fv/8);
           VVVV = VVVV - 3*(fv/8);
           HHVV_re = HHVV_re - 1*(fv/8);
       end
       
       % 如果重新计算的体散射贡献大于总功率SpanMax，则令表面散射与二次散射贡献为零,再重新计算体散射贡献
       if ((HHHH <= eps) || (VVVV <= eps))
           fs = 0; fd = 0;
           if (ratio2 <= -2)
               fv = (HHHH + 8*(fv/15)) + HVHV + (VVVV + 3*(fv/15));
           elseif (ratio2 > 2)
               fv = (HHHH + 3*(fv/15)) + HVHV + (VVVV + 8*(fv/15));
           else
               fv = (HHHH + 3*(fv/8)) + HVHV + (VVVV + 3*(fv/8));
           end
       % 如果重新计算的体散射贡献小于总功率SpanMax，则分别计算表面散射与二次散射贡献
       else
           % 首先对数据进行处理，这步是为了保证测量减去体散射矩阵之后的残余矩阵是半正定的（有点类似于NNED）
           if((HHVV_re * HHVV_re + HHVV_im * HHVV_im) > HHHH * VVVV)
               rtemp = HHVV_re * HHVV_re + HHVV_im * HHVV_im;
               HHVV_re = HHVV_re * sqrt((HHHH * VVVV) / (rtemp+ eps));
               HHVV_im = HHVV_im * sqrt((HHHH * VVVV )/ (rtemp+ eps));
           % 若已经是半正定的，则不需要进行处理
%            else
%                HHVV_re
%                HHVV_im
           end
           
           % 奇次散射占主导
           if (HHVV_re >=0)
               ALP_re = -1; ALP_im = 0;
               fd = (HHHH * VVVV - HHVV_re * HHVV_re - HHVV_im * HHVV_im) / (HHHH + VVVV + 2 * HHVV_re + eps);
               fs = VVVV - fd;
               BET_re = (fd + HHVV_re) / (fs+ eps);
               BET_im = HHVV_im / (fs+ eps);   
           % 偶次散射占主导    
           else              
               BET_re = 1; BET_im = 0;
               fs = (HHHH * VVVV - HHVV_re * HHVV_re - HHVV_im * HHVV_im) / (HHHH + VVVV - 2 * HHVV_re+ eps);
               fd = VVVV - fs;
               ALP_re = (HHVV_re - fs) / (fd+ eps);
               ALP_im = HHVV_im / (fd+ eps);
               
           end
       end
       
       % 求各分量功率
       Ps = fs * (1 + BET_re * BET_re + BET_im * BET_im);
       Pd = fd * (1 + ALP_re * ALP_re + ALP_im * ALP_im);
       Pv = fv;
       Pc = 0;
       
       if(Ps<SpanMin) Ps = SpanMin; elseif(Ps>SpanMax) Ps = SpanMax; end
       if(Pd<SpanMin) Pd = SpanMin; elseif(Pd>SpanMax) Pd = SpanMax; end
       if(Pv<SpanMin) Pv = SpanMin; elseif(Pv>SpanMax) Pv = SpanMax; end
       
       Podd(k) = Ps;
       Pdbl(k) = Pd;
       Pvol(k) = Pv;
       Phel(k) = Pc;
         
   % 体散射功率大于零，则进行四成分分解
   else
       % /********************************************/
       % /****     Yamaguchi四成分分解算法         ****/
       % /********************************************/
       % 若表面散射占主导
       if(HV_type == 1)
            S = TT11 - (PPvol / 2);
            D = TP - PPvol - PPhel - S;
            Cre = TT12_re + TT13_re;
            Cim = TT12_im + TT13_im;
            if (ratio1<= -2)
                Cre = Cre - (PPvol / 6);
            elseif (ratio1> 2)
                Cre = Cre + (PPvol / 6);
            end
            
            % 若体散射贡献与螺旋体贡献之和大于总功率,则为二成分分解
            if ((PPvol + PPhel) > TP)
                yPs = 0;
                yPd = 0;
                PPvol = TP - PPhel;
            else
                % CO是用以判断表面散射占主导，还是二次散射占主导
                CO = 2*TT11 + PPhel - TP;  % Singh一文中提出的，实际上是Yajima等人提出来的
                % 表面散射占主导,则alpha = 0
                if (CO > 0)
                    yPs = S + (Cre*Cre + Cim*Cim)/(S+eps);
                    yPd = D - (Cre*Cre + Cim*Cim)/(S+eps);
                % 二次散射占主导,则beta = 0
                else
                    yPd = D + (Cre*Cre + Cim*Cim)/(D+eps);
                    yPs = S - (Cre*Cre + Cim*Cim)/(D+eps);
                end
            end
            
            % 若表面散射贡献小于零
            if (yPs < 0)
                % 表面散射贡献小于零情况下若二次散射贡献也小于零（也退化为二成分分解）
                if (yPd < 0)
                    yPs = 0.; yPd = 0.;
                    PPvol = TP - PPhel;
                % 否则进行三成分分解    
                else
                    yPs = 0.;
                    yPd = TP - PPvol - PPhel;
                end
            % 表面散射贡献大于零，而二次散射贡献小于零    
            elseif (yPd < 0)
                yPd = 0.;
                yPs = TP - PPvol - PPhel;
            end
       
       % 二次散射占主导     
       elseif(HV_type == 2)
           
           S = TT11;
           D = TP - PPvol - PPhel - S;
           Cre = TT12_re + TT13_re;
           Cim = TT12_im + TT12_im;
           
           yPd = D + (Cre*Cre + Cim*Cim)/(D+eps);
           yPs = S - (Cre*Cre + Cim*Cim)/(D+eps);
           
            % 若表面散射贡献小于零
            if (yPs < 0)
               % 表面散射贡献小于零情况下若二次散射贡献也小于零（也退化为二成分分解）
               if (yPd < 0)
                   yPs = 0; yPd = 0;
                   PPvol = TP - PPhel;
               else
                   yPs = 0.;
                   yPd = TP - PPvol - PPhel;
               end
            % 表面散射贡献大于零，而二次散射贡献小于零    
            elseif (yPd < 0)
                yPd = 0;
                yPs = TP - PPvol - PPhel;  
            end
       end
       
       
       if(yPs<SpanMin) yPs = SpanMin; elseif(yPs>SpanMax) yPs = SpanMax; end
       if(yPd<SpanMin) yPd = SpanMin; elseif(yPd>SpanMax) yPd = SpanMax; end
       if(PPvol<SpanMin) PPvol = SpanMin; elseif(PPvol>SpanMax) PPvol = SpanMax; end
       if(PPhel<SpanMin) PPhel = SpanMin; elseif(PPhel>SpanMax) PPhel = SpanMax; end

       
       Podd(k) = yPs;
       Pdbl(k) = yPd;
       Pvol(k) = PPvol;
       Phel(k) = PPhel; 
   end

end

% 矢量矩阵化
Podd = reshape(Podd,M,N);
Pdbl = reshape(Pdbl,M,N);
Pvol = reshape(Pvol,M,N);
Phel = reshape(Phel,M,N);
end

