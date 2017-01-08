function [matching] = Hungarian(map)
% Hungarian
% Algorithm is from:
% http://www.public.iastate.edu/~ddoty/HungarianAlgorithm.html

    matching = zeros(size(map));

    x_con = find(sum(~isinf(map), 2) ~= 0);
    y_con = find(sum(~isinf(map), 1) ~= 0);
    
    C_size = max(length(x_con),length(y_con));
    C_mat = zeros(C_size);
    C_mat(1:length(x_con), 1:length(y_con)) = map(x_con, y_con);
    if isempty(C_mat)
        return
    end

    stepnum = 1;
    while true
        switch stepnum
          case 1
            C_mat = step1(C_mat);
            stepnum = 2;
          case 2
            [covered_row, covered_col, M] = step2(C_mat);
            stepnum = 3;
          case 3
            [covered_col, stepnum] = step3(M, C_size);
          case 4
            [M, covered_row, covered_col, Z_row, Z_col, stepnum] = step4(C_mat,covered_row,covered_col,M);
          case 5
            [M, covered_row, covered_col] = step5(M,Z_row,Z_col,covered_row,covered_col);
            stepnum = 3;
          case 6
            C_mat = step6(C_mat, covered_row, covered_col);
            stepnum = 4;
          case 7
            break;
        end
    end

    matching(x_con, y_con) = M(1:length(x_con), 1:length(y_con));
end

function C_mat = step1(C_mat)
    for i = 1:length(C_mat)
        min_val = min(C_mat(i, :));
        C_mat(i,:) = C_mat(i,:) - min_val;
    end
end
  
function [covered_row, covered_col, M] = step2(C_mat)
    n = length(C_mat);
    covered_row = zeros(n, 1);
    covered_col = zeros(n, 1);
    M = zeros(n, n);
    
    for i = 1:n
        for j = 1:n
            if C_mat(i,j) == 0 && covered_row(i) == 0 && covered_col(j) == 0
                M(i,j) = 1;
                covered_row(i) = 1;
                covered_col(j) = 1;
            end
        end
    end
    
    covered_row(:) = 0;
    covered_col(:) = 0;
end
    
function [covered_col,stepnum] = step3(M, C_size)
    covered_col = sum(M, 1);
    if sum(covered_col) == C_size
        stepnum = 7;
    else
        stepnum = 4;
    end
end
  
function [M, covered_row, covered_col, Z_row, Z_col, stepnum] = step4(C_mat, covered_row, covered_col, M)
    function [row, col] = find_a_zero() 
        row = 0;
        col = 0;
        i = 1;
        done2 = false;
        while ~done2
            j = 1;
            while true
                if C_mat(i,j) == 0 && covered_row(i) == 0 && covered_col(j) == 0
                    row = i;
                    col = j;
                    done2 = true;
                end
                j = j + 1;
                if j > n
                    break;
                end
            end
            i = i + 1;
            if i > n
                done2 = true;
            end
        end
    end

    n = length(C_mat);
    done = false;
    while ~done
        [row, col] = find_a_zero();
        if row == 0
            stepnum = 6;
            done = true;
            Z_row = 0;
            Z_col = 0;
        else
            M(row,col) = 2;
            if sum(find(M(row,:)==1)) ~= 0
                covered_row(row) = 1;
                zcol = find(M(row,:)==1);
                covered_col(zcol) = 0;
            else
                stepnum = 5;
                done = true;
                Z_row = row;
                Z_col = col;
            end            
        end
    end
end
    
function [M,covered_row,covered_col] = step5(M, Z_row, Z_col, covered_row, covered_col)
    done = false;
    count = 1;
    while ~done
        rindex = find(M(:,Z_col(count))==1);
        if rindex > 0
            count = count + 1;
            Z_row(count,1) = rindex;
            Z_col(count,1) = Z_col(count - 1);
        else
            done = true;
        end
        
        if ~done
            cindex = find(M(Z_row(count),:)==2);
            count = count + 1;
            Z_row(count, 1) = Z_row(count - 1);
            Z_col(count, 1) = cindex;    
        end    
    end
    
    for i = 1:count
        if M(Z_row(i), Z_col(i)) == 1
            M(Z_row(i), Z_col(i)) = 0;
        else
            M(Z_row(i), Z_col(i)) = 1;
        end
    end
    
    covered_row(:) = 0;
    covered_col(:) = 0;
    
    M(M == 2) = 0;
end
    
function C_mat = step6(C_mat, covered_row, covered_col)
    from_row = find(covered_row == 0);
    from_col = find(covered_col == 0);
    minval = min(min(C_mat(from_row, from_col)));
    
    C_mat(find(covered_row == 1), :) = C_mat(find(covered_row == 1), :) + minval;
    C_mat(:, find(covered_col == 0)) = C_mat(:,find(covered_col == 0)) - minval;
end
    
function cnum = min_line_cover(Edge)
    [covered_row, covered_col, M] = step2(Edge);
    covered_col = step3(M, length(Edge));
    [M, covered_row, covered_col, Z_row, Z_col] = step4(Edge,covered_row,covered_col,M);
    cnum = length(Edge) - sum(covered_row) - sum(covered_col);
end