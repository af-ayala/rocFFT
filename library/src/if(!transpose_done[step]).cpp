        for(size_t step = 0; step < 3; ++step)
        {
            if(!transpose_done[step])
            {

                obtain next_field
                
                // perform intermediate global transpositions but not yet the last one, because the last one
                // maybe already has bricks aligned as pencils, eg output grid 1 2 2
                if(currentField.bricks != nextField.bricks &&  nextField.bricks != desc.outFields.front().bricks)
                {
                    transpose current_field to next_field
                }

                compute fft along contiguos dimensions of next_field
           }

           if still missing one transpose, perform it 
           perform missing fft

