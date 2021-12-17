clear;figure(1);clf
ly = 1.0;
lx = 0.6;
for it = 50:50:2500
    load(['../out_vis/step_' int2str(it) '.mat'])
    tiledlayout(2,2,"TileSpacing","compact","Padding","compact")
    nexttile(1);imagesc(C')
    shading flat;axis image;colorbar;colormap(gca,copper);set(gca,"YDir","normal");title('C')
    xticks([]);yticks([])
    nexttile(2);imagesc(Pr')
    shading flat;axis image;colorbar;colormap(gca,turbo);set(gca,"YDir","normal");title('Pr')
    xticks([]);yticks([])
    nexttile(3);imagesc(Vx')
    shading flat;axis image;colorbar;colormap(gca,spring);set(gca,"YDir","normal");title('Vx')
    xticks([]);yticks([])
    nexttile(4);imagesc(Vy')
    shading flat;axis image;colorbar;colormap(gca,spring);set(gca,"YDir","normal");title('Vy')
    xticks([]);yticks([])
    sgtitle(it)
    drawnow
    frame = getframe(gcf); 
    im    = frame2im(frame); 
    [imind,cm] = rgb2ind(im,256); 
    % Write to the GIF File
    if it==50
        imwrite(imind,cm,'navier_stokes.gif','gif', 'Loopcount',inf,'DelayTime',1/8);
    else
        imwrite(imind,cm,'navier_stokes.gif','gif','WriteMode','append','DelayTime',1/8);
    end 
end