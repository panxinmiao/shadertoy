from shadertoy import Shadertoy

# https://www.shadertoy.com/view/MtfGWM

main_code = """
//another holy grail candidate from msltoe found here:
//http://www.fractalforums.com/theory/choosing-the-squaring-formula-by-location

//I have altered the formula to make it continuous but it still creates the same nice julias - eiffie

#define time iTime
#define size iResolution

vec3 C,mcol;
bool bColoring=false;
#define pi 3.14159
float DE(in vec3 p){
	float dr=1.0,r=length(p);
	//C=p;
	for(int i=0;i<10;i++){
		if(r>20.0)break;
		dr=dr*2.0*r;
		float psi = abs(mod(atan(p.z,p.y)+pi/8.0,pi/4.0)-pi/8.0);
		p.yz=vec2(cos(psi),sin(psi))*length(p.yz);
		vec3 p2=p*p;
		p=vec3(vec2(p2.x-p2.y,2.0*p.x*p.y)*(1.0-p2.z/(p2.x+p2.y+p2.z)),
			2.0*p.z*sqrt(p2.x+p2.y))+C;	
		r=length(p);
		if(bColoring && i==3)mcol=p;
	}
	return min(log(r)*r/max(dr,1.0),1.0);
}

float rnd(vec2 c){return fract(sin(dot(vec2(1.317,19.753),c))*413.7972);}
float rndStart(vec2 fragCoord){
	return 0.5+0.5*rnd(fragCoord.xy+vec2(time*217.0));
}
float shadao(vec3 ro, vec3 rd, float px, vec2 fragCoord){//pretty much IQ's SoftShadow
	float res=1.0,d,t=2.0*px*rndStart(fragCoord);
	for(int i=0;i<4;i++){
		d=max(px,DE(ro+rd*t)*1.5);
		t+=d;
		res=min(res,d/t+t*0.1);
	}
	return res;
}
vec3 Sky(vec3 rd){//what sky??
	return vec3(0.5+0.5*rd.y);
}
vec3 L;
vec3 Color(vec3 ro, vec3 rd, float t, float px, vec3 col, bool bFill, vec2 fragCoord){
	ro+=rd*t;
	bColoring=true;float d=DE(ro);bColoring=false;
	vec2 e=vec2(px*t,0.0);
	vec3 dn=vec3(DE(ro-e.xyy),DE(ro-e.yxy),DE(ro-e.yyx));
	vec3 dp=vec3(DE(ro+e.xyy),DE(ro+e.yxy),DE(ro+e.yyx));
	vec3 N=(dp-dn)/(length(dp-vec3(d))+length(vec3(d)-dn));
	vec3 R=reflect(rd,N);
	vec3 lc=vec3(1.0,0.9,0.8),sc=sqrt(abs(sin(mcol))),rc=Sky(R);
	float sh=clamp(shadao(ro,L,px*t,fragCoord)+0.2,0.0,1.0);
	sh=sh*(0.5+0.5*dot(N,L))*exp(-t*0.125);
	vec3 scol=sh*lc*(sc+rc*pow(max(0.0,dot(R,L)),4.0));
	if(bFill)d*=0.05;
	col=mix(scol,col,clamp(d/(px*t),0.0,1.0));
	return col;
}
mat3 lookat(vec3 fw){
	fw=normalize(fw);vec3 rt=normalize(cross(fw,vec3(0.0,1.0,0.0)));return mat3(rt,cross(rt,fw),fw);
}

vec3 Julia(float t){
	t=mod(t,5.0);
	if(t<1.0)return vec3(-0.8,0.0,0.0);
	if(t<2.0)return vec3(-0.8,0.62,0.41);
	if(t<3.0)return vec3(-0.8,1.0,-0.69);
	if(t<4.0)return vec3(0.5,-0.84,-0.13);
	return vec3(0.0,1.0,-1.0);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
	float px=0.5/size.y;
	L=normalize(vec3(0.4,0.8,-0.6));
	float tim=time*0.5;
	
	vec3 ro=vec3(cos(tim*1.3),sin(tim*0.4),sin(tim))*3.0;
	vec3 rd=lookat(vec3(-0.1)-ro)*normalize(vec3((2.0*fragCoord.xy-size.xy)/size.y,3.0));
	
	tim*=0.6;
	if(mod(tim,15.0)<5.0)C=mix(Julia(tim-1.0),Julia(tim),smoothstep(0.0,1.0,fract(tim)*5.0));
	else C=vec3(-cos(tim),cos(tim)*abs(sin(tim*0.3)),-0.5*abs(-sin(tim)));

	float t=DE(ro)*rndStart(fragCoord),d=0.0,od=10.0;
	vec3 edge=vec3(-1.0);
	bool bGrab=false;
	vec3 col=Sky(rd);
	for(int i=0;i<78;i++){
		t+=d*0.5;
		d=DE(ro+rd*t);
		if(d>od){
			if(bGrab && od<px*t && edge.x<0.0){
				edge=vec3(edge.yz,t-od);
				bGrab=false;
			}
		}else bGrab=true;
		od=d;
		if(t>10.0 || d<0.00001)break;
	}
	bool bFill=false;
	d*=0.05;
	if(d<px*t && t<10.0){
		if(edge.x>0.0)edge=edge.zxy;
		edge=vec3(edge.yz,t);
		bFill=true;
	}
	for(int i=0;i<3;i++){
		if(edge.z>0.0)col=Color(ro,rd,edge.z,px,col,bFill,fragCoord);
		edge=edge.zxy;
		bFill=false;
	}
	fragColor = vec4(2.0*col,1.0);
}

"""

sound_code = """
#define bps 6.0
float nofs(float n){//the song's "random" ring
    n=mod(n,8.0);
    if(n<1.0)return 0.0;
    if(n<2.0)return 1.0;
    if(n<3.0)return 2.0;
    if(n<4.0)return 3.0;
    if(n<5.0)return -3.0;
    if(n<6.0)return -2.0;
    if(n<7.0)return -1.0;
    return 0.0;
}

float scale(float note){//throws out dissonant tones
	float n2=mod(note,12.0);
	//if((n2==1.0)||(n2==3.0)||(n2==6.0)||(n2==8.0)||(n2==10.0))note=-100.0;//major
	//if((n2==1.0)||(n2==4.0)||(n2==6.0)||(n2==9.0)||(n2==11.0))note=-100.0;//minor
	if((n2==1.0)||(n2==4.0)||(n2==5.0)||(n2==9.0)||(n2==10.0))note=-100.0;//hungarian minor
	if(note>84.0)note=84.0+n2;
	return note;
}
#define TAO 6.283185
// note number to frequency  from https://www.shadertoy.com/view/ldfSW2
//float ntof(float n){if(n<12.0)return 0.0;return 440.0 * pow(2.0, (n - 67.0) / 12.0);}

float ntof(float note){//note frequencies from wikipedia
	if(note<12.0)return 0.0;
	float octave=floor((note+0.5)/12.0)-5.0;
	note=mod(note,12.0);
	float nt=493.88;
    if(note<0.5)nt=261.63;
	else if(note<1.5)nt=277.18;
	else if(note<2.5)nt=293.66;
    else if(note<3.5)nt=311.13;
    else if(note<4.5)nt=329.63;
    else if(note<5.5)nt=349.23;
    else if(note<6.5)nt=369.99;
    else if(note<7.5)nt=392.0;
    else if(note<8.5)nt=415.30;
    else if(note<9.5)nt=440.0;
    else if(note<10.5)nt=466.16;
	return nt*pow(2.0,octave);
}


float ssaw(float t){return 4.0*abs(fract(t)-0.5)-1.0;}
float rnd(float t){return fract(sin(t*341.545234)*1531.2341);}
float srnd(float t){float t2=fract(t);return mix(rnd(floor(t)),rnd(floor(t+1.0)),t2*t2*(3.0-2.0*t2));}
float harm(float x,float ps,float hm,float sp){//phase shift, harmonics, spacing
	float a2=0.0,s=1.0;
	for(int i=0;i<10;i++){
		if(i<int(hm)){
			a2+=sin((x*s+ps)*TAO)/s;
			s+=sp;
		}
	}
	return a2*0.5;
}
vec2 inst(float n,float t,float bt,float pan,int i){
	float f=ntof(scale(n)),ps=0.0,hm=0.0,sp=1.0;
	if(f<12.0)return vec2(0.0);	
	if(i==0){ps=pow(bt*0.5,0.25)*0.2;hm=9.0;}
	else if(i==1){ps=bt*0.5;hm=4.0;sp=0.5;}
	else if(i==9){ps=bt*rnd(t);hm=10.0-4.0*bt;f*=0.5+0.5*rnd(t);}
	float a=harm(f*t,ps,hm,sp);
	a*=exp(-bt*(0.9+float(i)))*min(min(bt,2.0-bt)*100.0,1.0)*60.0/n;
	return vec2(a*(1.0-pan),a*pan);
}
vec2 inst2(float nn,float no,float of,float t,float bt,float pan,int i){
	return inst(nn+of,t,bt,pan,i)+inst(no+of,t,bt+1.0,pan,i);//plays new note and tail of last note
}
vec2 mainSound( in int samp,float time)
{
	float tim=time*bps;
	if(tim>128.0 && tim<256.0)tim=224.0-tim;
	float b=floor(tim);
	float t0=fract(tim),t1=mod(tim,2.0)*0.5,t2=mod(tim,4.0)*0.25;
	float n2=nofs(b*0.0625)+nofs(b*0.125)+nofs(b*0.25);
	float n1=n2+nofs(b*0.5),n0=n1+nofs(b);
	b-=1.0;//go back in time to finish old notes
	float n5=nofs(b*0.0625)+nofs(b*0.125)+nofs(b*0.25);
	float n4=n5+nofs(b*0.5),n3=n4+nofs(b);
	vec2 a0=inst2(n0,n3,72.0,time,t0,0.8,0);
	b-=1.0;
	n5=nofs(b*0.0625)+nofs(b*0.125)+nofs(b*0.25);
	n4=n5+nofs(b*0.5);
	vec2 a1=inst2(n1,n4,60.0,time,t1,0.5,0);
	vec2 a1h=inst2(n1,n4,57.0,time,t1,0.6,0);
	b-=2.0;
	n5=nofs(b*0.0625)+nofs(b*0.125)+nofs(b*0.25);
	vec2 a2=inst2(n2,n5,36.0,time,t2,0.2,0);
	//vec2 a2h=vec2(0.0);//inst2(n2,n5,53.0,time,t2,0.25,0);
	//vec2 a1hb=inst(n1+64.0,time,t1,0.0,1)*2.0;
	vec2 v=0.25*(a0+a1+a1h+a2);
	return clamp(v,-1.0,1.0);
}
"""

if __name__ == "__main__":
    shader = Shadertoy(main_code, sound_code=sound_code)
    shader.show()
