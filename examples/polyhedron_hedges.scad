module my_sphere(P1, radius){
	x1 = P1[0]; y1 = P1[1]; z1 = P1[2];	
	
	translate([x1, y1, z1])
		sphere(r=radius);
}
module bar(P1, P2, radius) {
	x1 = P1[0]; y1 = P1[1]; z1 = P1[2];
	x2 = P2[0]; y2 = P2[1]; z2 = P2[2];
	Dx = x2-x1; Dy = y2-y1; Dz = z2-z1;
	
	Length = sqrt(pow(Dx, 2) + pow(Dy, 2) + pow(Dz, 2));
	b = acos(Dz/Length);
	c = (Dx==0) ? sign(Dy)*90 : ( (Dx>0) ? atan(Dy/Dx) : atan(Dy/Dx)+180 );
	
	union(){
	translate([x1, y1, z1])
	rotate([0, b, c])
		cylinder(h=Length, r=radius);

	my_sphere([x1, y1, z1], radius);
	my_sphere([x2, y2, z2], radius);
	}
}
module polyhedron_hedges(nodes, edges, radius){
    nnodes = len(nodes);
    nedges =len(edges);

    for (i=[0:nedges-1]){
        P1 = nodes[edges[i][0]-1];
        P2 = nodes[edges[i][1]-1];
        bar(P1, P2, radius);
    }

}
module rect_edge(P1 = [0, 0, 0], P2=[1, 0, 0], d=0.2, t=0.2) {
	x1 = P1[0]; y1 = P1[1]; z1 = P1[2];
	x2 = P2[0]; y2 = P2[1]; z2 = P2[2];
	Dx = x2-x1; Dy = y2-y1; Dz = z2-z1;
	
	Length = sqrt(pow(Dx, 2) + pow(Dy, 2) + pow(Dz, 2));
	b = acos(Dz/Length);
	c = (Dx==0) ? sign(Dy)*90 : ( (Dx>0) ? atan(Dy/Dx) : atan(Dy/Dx)+180 );
	
   	translate([x1, y1, z1])
	rotate([0, b, c])
	translate([-d/2, -t/2, 0])
	cube(size=[d, t, Length], center = false);
	
	translate([x1, y1, z1]) cylinder(h=d, r=t/2, center = true);
	translate([x2, y2, z2]) cylinder(h=d, r=t/2, center = true);
	//my_sphere([x1, y1, z1], radius);
	//my_sphere([x2, y2, z2], radius);
}
module wall(P1 = [0, 0], P2=[1, 0], w=0.2, h=0.2) {

x1 = P1[0]; y1 = P1[1]; 
x2 = P2[0]; y2 = P2[1]; 
Dx = x2-x1; Dy = y2-y1; 

L = sqrt(pow(Dx, 2) + pow(Dy, 2));
phi = (Dx==0) ? sign(Dy)*90 : ( (Dx>0) ? atan(Dy/Dx) : atan(Dy/Dx)+180 );

translate(P1) linear_extrude(h) rotate([0, 0, phi])
{
    circle(w/2);
    translate([0, -w/2]) square([L, w]);
    translate([L, 0]) circle(w/2);
};

}
