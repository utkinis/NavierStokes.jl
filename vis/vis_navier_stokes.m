clear;figure(1);clf
ly = 1.0;
lx = 0.6;
iframe = 0;
for it = 10:10:2000
    load(['../scripts/out_vis/step_' int2str(it) '.mat'])
    if it == 10
        [xc,yc] = ndgrid(1:size(Pr,1),1:size(Pr,2));
    end
    om = diff(Vy(:,2:end-1),1,1)/dx - diff(Vx(2:end-1,:),1,2)/dy;
    tiledlayout(2,2,"TileSpacing","compact","Padding","compact")
    Vxc = 0.5*(Vx(1:end-1,:)+Vx(2:end,:));
    Vyc = 0.5*(Vy(:,1:end-1)+Vy(:,2:end));
    Vmag = sqrt(Vxc.^2 + Vyc.^2);
    nexttile(1);imagesc(C')
    shading flat;axis image;colorbar;colormap(gca,copper);set(gca,"YDir","normal");title('\itc')
    xticks([]);yticks([])
    nexttile(2);imagesc(om');caxis([-100 100])
    shading flat;axis image;colorbar;colormap(gca,cool);set(gca,"YDir","normal");title('\omega')
    xticks([]);yticks([])
    nexttile(3);imagesc(Pr');caxis([-1 0.5])
    shading flat;axis image;colorbar;colormap(gca,jet);set(gca,"YDir","normal");title('\itp')
    hold on
    st = 20;
    quiver(xc(1:st:end,1:st:end),yc(1:st:end,1:st:end),Vxc(1:st:end,1:st:end),Vyc(1:st:end,1:st:end),1.0,'filled','LineWidth',1,'Color','w')
    hold off
    axis([1 size(Pr,1) 1 size(Pr,2)])
    xticks([]);yticks([])
    nexttile(4);imagesc(Vmag');caxis([0 1.5])
    shading flat;axis image;colorbar;colormap(gca,summer);set(gca,"YDir","normal");title('|{\itv}|')
    xticks([]);yticks([])
    sgtitle(it)
    drawnow
    exportgraphics(gcf,sprintf("anim/frame_%04d.png",iframe),'Resolution',300)
    iframe = iframe + 1;
end