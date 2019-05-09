function Aij=Aij_matrix(arguments)
    Aij = arguments(1);
    k = arguments(2);
    pos_i=arguments(3);
    pos_j =arguments(4);

    temp = pos_j - pos_i;
    r_ij = sqrt(temp(1)^2+temp(2)^2+temp(3)^2);
    % calculate the unit vectors between two particles
    nx, ny, nz = temp / r_ij;
    eikr = exp(1i * k * r_ij);

    A = (k ^ 2) * eikr / r_ij;
    B = (1 / r_ij ^ 3 - 1i * k / r_ij ^ 2) * eikr;
    C = 3*B-A;

    Aij(1,1) = A * (ny ^ 2 + nz ^ 2) + B * (3 * nx ^ 2 - 1);
    Aij(2,2) = A * (nx ^ 2 + nz ^ 2) + B * (3 * ny ^ 2 - 1);
    Aij(3,3) = A * (ny ^ 2 + nx ^ 2) + B * (3 * nz ^ 2 - 1);

    Aij(1,2) = Aij(2,1) = nx * ny * C;
    Aij(1,3) = Aij(3,1) = nx * nz * C;
    Aij(1,2) = Aij(2,1) = ny * nz * C;
end

function A=calc_A_matrix(A, k, N, r_eff, epsilon, eps_surround,pos)
    Aij = zeros(3, 3); % interaction matrix
    if N==1
		A=inv(calc_alpha_i(r_eff(i), epsilon, eps_surround,k));
    else
        for i =1:N
            for j =1:N
                if i == j
                    % diagonal of the matrix
                    A(3 *(i-1)+1: 3 * i) = inv(calc_alpha_i(r_eff(i), epsilon(i), eps_surround,k));
                else
                    A(3 *(i-1)+1: 3 * i) = -Aij_matrix([Aij, k, pos(i), pos(j)]);
				end
			end
		end
	end
end
	
function E_inc=calc_E_inc(k_vec_k, N, pos, E0, E_inc):
    for i =1:N
        E_inc(3*(i-1)+1:3*i) = E0(3*(i-1)+1:3*i) * exp(1i * k_vec_k * pos(i)));
	end
end
	
function [q_ext,q_abs,q_scat,p_calc,c_extind,c_absind,field_enh]=cdacalc(wave,epsvec,eps_surround,N,pos,r_eff,E0_vec,k_vec)
    % disp some comforting stuff for the user
    n_wave = numel(wave);
    disp("PyCDA by K.C.")
    disp("Input data:")
    disp(["Number of particles: ",num2str(N)])
    disp(["Number of wavelengths: ",num2str(n_wave)])
    
    % pre allocate the arrays/vectors
    A = zeros(3 * N, 3 * N); % A matrix of 3N x 3N filled with zeros
    p = zeros(3 * N); % A vector that stores polarization Px, Py, Pz of the each particles, we will use this for initial guess for solver
    E0 = repmat(E0_vec, 1,N); % A vector that has the Ex, Ey, Ez for each particles, we make this by tiling the E0_vec vector
    E_inc = zeros(3 * N);% A vector storing the Incident Electric field , Einc_x, Einc_y, Einc_z for each particle

    p_calc = zeros(3 * N, n_wave);  % This stores the dipoles moments for each particle at different wavelengths
    c_ext = zeros(n_wave); % stores the extinction crosssection
    c_abs = zeros(n_wave); % stores the absorption crossection
    c_scat = c_ext-c_abs; % stores the scattering crossection
    c_absind= zeros(N,n_wave); % stores the absorption crosssection of individual particles
    c_extind=zeros(N,n_wave);
    field_enh=zeros(N,n_wave);

    % loop over the wavelength
    for w_index=1:numel(wave)
		w=wave(w_index);
        start_at_this_wavelength = toc()
        disp(['Running wavelength: ', num2str(w * 1E9)])
        k = (2*pi/w)
        if N==1
            epsilon=epsvec(w_index);
        else
            epsilon = epsvec(:,w_index); % Get the dielectric function    
		end
        A=calc_A_matrix(A, k, N, r_eff, epsilon, eps_surround,pos) % Calculate the A matrix
        E_inc=calc_E_inc(k_vec*k, N, pos, E0, E_inc)
        % We will functionine a callback that calculates the current residual
        iter = 1
		indvec=(3*(i-1)+1:3*i,3*(i-1)+1:3*i);
        disp("Iteration : Residual")
        p_calc(:,w_index) =  dsl.spsolve(A, E_inc, use_umfpack=False)
        info=0;
        if info == 0
            disp('Successful Exit')
            % calculate the extinction crossection
            c_ext(w_index) = (4 * pi * k / norm(E0) ^ 2) * sum(imag(dot(conj(E_inc),p_calc(:,w_index))));
            % calculate the absorption crossection
            for i =1:N
				pind=p_calc(indvec,w_index);
				Aind=A(indvec);
                c_absind(i,w_index)=(4*pi*k/norm(E0)^2)*(imag(dot(pind,dot(conj(Aind,conj(pind)))))-(2.0/3)*k^3*norm(pind^2));
                c_extind(i,w_index)= (4*pi*k/norm(E0)^2)*(imag(dot(pind,dot(conj(Aind,conj(pind)))-(0.0/3)*k^3*norm(pind)^2)));
                c_abs(w_index) = c_abs(w_index)+ (imag(dot(pind,dot(conj(Aind),conj(pind))))-(2.0/3)*k^3*norm(pind^2));
			end	
            c_abs(w_index) = c_abs(w_index)*(4*pi*k/norm(E0)^2);
            c_scat(w_index) = c_ext(w_index)-c_abs(w_index);
            
            if N==2
                As=-A(1:2,3:6);
                i=0;
                alf=A(6,6)^(-1); %get polarizability, each incident field, and scattered field (without coupling) from the other sphere
                %A[3*(i + 1):3*(i + 2), 3*i:3*(i + 1)]
                field_enh(i,w_index)=(norm(E_inc(3*(i-1)+1 : 3*i)+alf*dot(As,E_inc(3*i+1:3*(i + 1))))^2;
                i=1;
                alf=A(1,1)^(-1);
                disp(alf);
                disp(As);
                field_enh(i,w_index)=(norm(E_inc(3*(i-1)+1 : 3*i)+alf*dot(As,E_inc(3*(i - 1)+1:3*i))/norm(E_inc(3*(i-1)+1 : 3*i)))^2;
			end
        elseif info > 0
            disp('Convergence not achieved, may be increase the number of maxiter')
        elseif info < 0
            disp(info)
            disp('illegal input')
        end
        end_at_this_wavelength = toc()
	end
    q_ext = efficiency_calc(c_ext, r_eff);
    q_abs = efficiency_calc(c_abs, r_eff);
    q_scat = efficiency_calc(c_scat, r_eff);
end