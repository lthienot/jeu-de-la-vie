__kernel void transpose_naif (__global unsigned *in, __global unsigned *out)
{
  int x = get_global_id (0);
  int y = get_global_id (1);

  out [x * DIM + y] = in [y * DIM + x];
}



__kernel void transpose (__global unsigned *in, __global unsigned *out)
{
  __local unsigned tile [TILEX][TILEY+1];
  int x = get_global_id (0);
  int y = get_global_id (1);
  int xloc = get_local_id (0);
  int yloc = get_local_id (1);

  tile [xloc][yloc] = in [y * DIM + x];

  barrier (CLK_LOCAL_MEM_FENCE);

  out [(x - xloc + yloc) * DIM + y - yloc + xloc] = tile [yloc][xloc];
}

//jeu de la vie naif
__kernel void life_naif (__global unsigned* in, __global unsigned* out)
{
  int x = get_global_id(0);
  int y = get_global_id(1);

  int count = 0;

  if (x > 0 && x < DIM - 1 && y > 0 && y < DIM - 1)
  {
  if(in[(y-1)*DIM+x-1])
    count++;
  if(in[(y-1)*DIM+x])
    count++;
  if(in[(y-1)*DIM+x+1])
    count++;
  if(in[y*DIM+x-1])
    count++;
  if(in[y*DIM+x+1])
    count++;
  if(in[(y+1)*DIM+x-1])
    count++;
  if(in[(y+1)*DIM+x])
    count++;
  if(in[(y+1)*DIM+x+1])
    count++;
  }
  if ((in[y*DIM+x] && (count < 2 || count > 3)) || (in[y*DIM+x] == 0 && (count !=3)))
    out[y*DIM+x] = 0;
  else
    out[y*DIM+x] = 0xFFFF00FF;
}

//jeu de la vie pas naif
__kernel void life (__global unsigned* in, __global unsigned* out)
{
  __local unsigned tile [TILEY+2][TILEX+2];
   int x = get_global_id(0);
   int y = get_global_id(1);
   int xloc = get_local_id(0);
   int yloc = get_local_id(1);
   int count = 0; //nb de voisins vivants

  /* remplissage de la tuile */
   tile[yloc+1][xloc+1] = in[y * DIM + x];
   if (xloc == 0 && x > 0) // bord gauche
      tile[yloc + 1][xloc] = in[y * DIM + x - 1];
   if (yloc == 0 && y > 0) //haut
      tile[yloc][xloc + 1] = in[(y - 1) * DIM + x];
   if (xloc == TILEX - 1 && x < DIM-1) //bord droit
      tile[yloc+1][TILEX + 1] = in[y * DIM + x + 1];
   if (yloc == TILEY-1 && y < DIM-1) //bas
      tile[TILEY + 1][xloc+1] = in[(y + 1) * DIM + x];
    if (xloc == 0 && x > 0 && yloc == 0 && y > 0) //coin haut gauche
      tile[yloc][xloc] = in[(y-1) * DIM + x - 1];
    if (xloc == 0 && x > 0 && yloc == TILEY-1 && y < DIM-1) //coin bas gauche
      tile[TILEY + 1][xloc] = in[(y+1) * DIM + x - 1];
    if (xloc == TILEX-1 && x < DIM-1 && yloc == 0 && y > 0) //coin haut droit
      tile[yloc][TILEX+1] = in[(y-1) * DIM + x + 1];
    if (xloc == TILEX-1 && x < DIM-1 && yloc == TILEY-1 && y < DIM-1) //coin bas droit
      tile[TILEY + 1][TILEX+1] = in[(y+1) * DIM + x + 1];

   barrier(CLK_LOCAL_MEM_FENCE);

  /* decompte du nb de voisins vivants */
   if (x > 0 && x < DIM - 1 && y > 0 && y < DIM - 1)
   {
     if(tile[yloc][xloc])
       count++;
     if(tile[yloc][xloc+1])
       count++;
     if(tile[yloc][xloc+2])
       count++;
     if(tile[yloc+1][xloc])
       count++;
     if(tile[yloc+1][xloc+2])
       count++;
     if(tile[yloc+2][xloc])
       count++;
     if(tile[yloc+2][xloc+1])
       count++;
     if(tile[yloc+2][xloc+2])
       count++;
   }

   /*on decide si on meurt ou on vit*/
   if ((tile[yloc+1][xloc+1] && (count < 2 || count > 3)) || (tile[yloc+1][xloc+1] == 0 && (count !=3)))
     out[y*DIM+x] = 0;
   else
     out[y*DIM+x] = 0xFFFF00FF;
}

//jeu de la vie pas naif
__kernel void life_stop (__global unsigned* in, __global unsigned* out, __global unsigned* stop)
{
  __local unsigned tile [TILEY+2][TILEX+2];
   int x = get_global_id(0);
   int y = get_global_id(1);
   int xloc = get_local_id(0);
   int yloc = get_local_id(1);
   int count = 0; //nb de voisins vivants

  /* remplissage de la tuile */
   tile[yloc+1][xloc+1] = in[y * DIM + x];
   if (xloc == 0 && x > 0) // bord gauche
      tile[yloc + 1][xloc] = in[y * DIM + x - 1];
   if (yloc == 0 && y > 0) //haut
      tile[yloc][xloc + 1] = in[(y - 1) * DIM + x];
   if (xloc == TILEX - 1 && x < DIM-1) //bord droit
      tile[yloc+1][TILEX + 1] = in[y * DIM + x + 1];
   if (yloc == TILEY-1 && y < DIM-1) //bas
      tile[TILEY + 1][xloc+1] = in[(y + 1) * DIM + x];
    if (xloc == 0 && x > 0 && yloc == 0 && y > 0) //coin haut gauche
      tile[yloc][xloc] = in[(y-1) * DIM + x - 1];
    if (xloc == 0 && x > 0 && yloc == TILEY-1 && y < DIM-1) //coin bas gauche
      tile[TILEY + 1][xloc] = in[(y+1) * DIM + x - 1];
    if (xloc == TILEX-1 && x < DIM-1 && yloc == 0 && y > 0) //coin haut droit
      tile[yloc][TILEX+1] = in[(y-1) * DIM + x + 1];
    if (xloc == TILEX-1 && x < DIM-1 && yloc == TILEY-1 && y < DIM-1) //coin bas droit
      tile[TILEY + 1][TILEX+1] = in[(y+1) * DIM + x + 1];

   barrier(CLK_LOCAL_MEM_FENCE);

  /* decompte du nb de voisins vivants */
   if (x > 0 && x < DIM - 1 && y > 0 && y < DIM - 1)
   {
     if(tile[yloc][xloc])
       count++;
     if(tile[yloc][xloc+1])
       count++;
     if(tile[yloc][xloc+2])
       count++;
     if(tile[yloc+1][xloc])
       count++;
     if(tile[yloc+1][xloc+2])
       count++;
     if(tile[yloc+2][xloc])
       count++;
     if(tile[yloc+2][xloc+1])
       count++;
     if(tile[yloc+2][xloc+2])
       count++;
   }

   /*on decide si on meurt ou on vit*/
   if ((tile[yloc+1][xloc+1] && (count < 2 || count > 3)) || (tile[yloc+1][xloc+1] == 0 && (count !=3)))
     out[y*DIM+x] = 0;
   else
     out[y*DIM+x] = 0xFFFF00FF;
  if (tile[yloc+1][xloc+1] != out[y*DIM+x])
    stop[0] = 0;

}



// NE PAS MODIFIER
static unsigned color_mean (unsigned c1, unsigned c2)
{
  uchar4 c;

  c.x = ((unsigned)(((uchar4 *) &c1)->x) + (unsigned)(((uchar4 *) &c2)->x)) / 2;
  c.y = ((unsigned)(((uchar4 *) &c1)->y) + (unsigned)(((uchar4 *) &c2)->y)) / 2;
  c.z = ((unsigned)(((uchar4 *) &c1)->z) + (unsigned)(((uchar4 *) &c2)->z)) / 2;
  c.w = ((unsigned)(((uchar4 *) &c1)->w) + (unsigned)(((uchar4 *) &c2)->w)) / 2;

  return (unsigned) c;
}

// NE PAS MODIFIER
static int4 color_to_int4 (unsigned c)
{
  uchar4 ci = *(uchar4 *) &c;
  return convert_int4 (ci);
}

// NE PAS MODIFIER
static unsigned int4_to_color (int4 i)
{
  return (unsigned) convert_uchar4 (i);
}



// NE PAS MODIFIER
static float4 color_scatter (unsigned c)
{
  uchar4 ci;

  ci.s0123 = (*((uchar4 *) &c)).s3210;
  return convert_float4 (ci) / (float4) 255;
}

// NE PAS MODIFIER: ce noyau est appelé lorsqu'une mise à jour de la
// texture de l'image affichée est requise
__kernel void update_texture (__global unsigned *cur, __write_only image2d_t tex)
{
  int y = get_global_id (1);
  int x = get_global_id (0);
  int2 pos = (int2)(x, y);
  unsigned c;

  c = cur [y * DIM + x];

  write_imagef (tex, pos, color_scatter (c));
}
