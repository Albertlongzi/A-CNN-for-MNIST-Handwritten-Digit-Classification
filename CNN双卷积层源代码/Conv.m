function y = Conv(x, W)
    [xr, xc, xch] = size(x);
    [wr, wc, wp, numF] = size(W);
    if xch>1&&(xch==wp)
        y = zeros(xr - wr + 1, xc - wc + 1, numF);
        W = W(:,:,end:-1:1,:);
        for i = 1:numF
            for j = 1:wp
                W(:,:,j,i) = rot90(W(:,:,j,i),2);
            end
            y(:, :, i) = convn(x, W(:,:,:,i), 'valid');
        end
    else
        y = zeros(xr - wr + 1, xc - wc + 1, wp);
        for k = 1:wp
            y(:, :, k) = conv2(x, rot90(W(:, :, k),2), 'valid');
        end
    end
    
end

