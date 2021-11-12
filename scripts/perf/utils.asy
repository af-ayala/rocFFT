// Find the comma-separated strings to use in the legend
string[] set_legends(string runlegs)
{
   string[] legends;
   bool myleg=((runlegs== "") ? false: true);
   bool flag=true;
   int n=-1;
   int lastpos=0;
   string legends[];
   if(myleg) {
      string runleg;
      while(flag) {
	 ++n;
	 int pos=find(runlegs,",",lastpos);
	 if(lastpos == -1) {runleg=""; flag=false;}
    
	 runleg=substr(runlegs,lastpos,pos-lastpos);

	 lastpos=pos > 0 ? pos+1 : -1;
	 if(flag) legends.push(runleg);
      }
   }
   return legends;
}

// Create an array from a comma-separated string
string[] listfromcsv(string input)
{
    string list[] = new string[];
    int n = -1;
    bool flag = true;
    int lastpos;
    while(flag) {
        ++n;
        int pos = find(input, ",", lastpos);
        string found;
        if(lastpos == -1) {
            flag = false;
            found = "";
        }
        found = substr(input, lastpos, pos - lastpos);
        if(flag) {
            list.push(found);
            lastpos = pos > 0 ? pos + 1 : -1;
        }
    }
    return list;
}


// Read the data from the output files generated by alltime.py.
void readfiles(string[] filelist, pair[][] xyval, pair[][] ylowhigh, bool secondary)
{
    for(int n = 0; n < filelist.length; ++n)
    {
        string filename = filelist[n];
        file fin = input(filename).line().word();
        string[] hdr = fin;
        bool moretoread = true;
        while(moretoread) {
            string label = fin;
            real xval = fin; // elements
            if(eof(fin)) {
                moretoread = false;
                break;
            }
            real yval = fin; // median
            xyval[n].push((xval, yval));

            real ylow = fin;
            real yhigh = fin;
            ylowhigh[n].push((ylow, yhigh));

            if(secondary) {
              real pval = fin;
            }
        }
    }
}

// Given an array of values, get the x and y min and max.
real[] xyminmax( pair[][] xyval )
{
  // Find the bounds on the data to determine if the scales should be
  // logarithmic.
  real xmin = inf;
  real xmax = 0.0;
  real ymin = inf;
  real ymax = 0.0;
  for(int i = 0; i < xyval.length; ++i) {
    for(int j = 0; j < xyval[i].length; ++j) {
      xmax = max(xmax, xyval[i][j].x);
      ymax = max(ymax, xyval[i][j].y);
      xmin = min(xmin, xyval[i][j].x);
      ymin = min(ymin, xyval[i][j].y);
    }
  }
  real[] vals;
  vals.push(xmin);
  vals.push(xmax);
  vals.push(ymin);
  vals.push(ymax);
  return vals;
}
