clear all
close all
clc
%rootpath is the path of the folder with subclasses
Rootpath='/Users/ninatubauribera/Desktop/groups/';
d=dir(Rootpath);
isub = [d(:).isdir];
nameFolds = {d(isub).name}';
nameFolds(ismember(nameFolds,{'.','..'})) = [];
nameFolds = char(nameFolds)
num_folders = 7

for j = 1 : num_folders
Datapath = strcat(Rootpath,nameFolds(j,:),'/')
FileNames = dir(strcat(Datapath,'*.vtk'));
%FileNames = dir(Datapath);

    for i = 1 : length(FileNames)
       
       [vertex,face] = read_vtk(strcat(Datapath,FileNames(i).name));
        X_coord = vertex(1,:)';
        Y_coord = vertex(2,:)';
        Z_coord = vertex(3,:)';

        %Surface = SurfStatReadSurf(strcat(Datapath,FileNames(i).name));
        %X = Surface.coord(1,:);
        shape.X = X_coord;
        %Y = Surface.coord(2,:);
        shape.Y = Y_coord;
        %Z = Surface.coord(3,:);
        shape.Z = Z_coord;
        %TRIV = Surface.tri;
        %shape{i}.TRIV = double(TRIV);
        shape.TRIV = double(face');
        Name = FileNames(i).name;
        shape.name = Name(:,1:end-4);

        K = 100;            % number of eigenfunctions
        alpha = 2;          % log scalespace basis

        T1 = [5:0.5:16];    % time scales for HKS
        T2 = [1:0.2:20];    % time scales for SI-HKS
        Omega = 2:20;       % frequencies for SI-HKS

        % compute cotan Laplacian
        [shape.W, shape.A] = mshlp_matrix(shape);
        shape.A = spdiags(shape.A,0,size(shape.A,1),size(shape.A,1));

        % compute eigenvectors/values
        [shape.evecs,shape.evals] = eigs(shape.W,shape.A,K,'SM');
        shape.evals = -diag(shape.evals);

        % compute descriptors
        shape.hks   = hks(shape.evecs,shape.evals,alpha.^T1);
        [shape.sihks, shape.schks] = sihks(shape.evecs,shape.evals,alpha,T2,Omega);
        
        save(strcat(Datapath,FileNames(i).name(1:end-4),'.mat'),'shape')

    end


    

end
%dataset=[shape{1}.X shape{1}.Y shape{1}.Z];
%[coeff,score,latent,tsquared,explained,mu]=pca(dataset);
%reconstruct=score*coeff'+repmat(mu,1002,1);

%diff=abs(shape{1,1}.sihks)-abs(shape{2,1}.sihks);

%figure(1)
%subplot(121)
%plot3(shape{1,1}.X,shape{1,1}.Y,shape{1,1}.Z,'.')
%hold on;
%subplot(122)
%plot3(X_a,Y_a,Z_a,'.')
%plot3(reconstruct(:,1),reconstruct(:,2),reconstruct(:,3),'.')

% hold on;
% plot3(shape{2,1}.X,shape{2,1}.Y,shape{2,1}.Z,'.')
% plot3(shape{3,1}.X,shape{3,1}.Y,shape{3,1}.Z,'.')