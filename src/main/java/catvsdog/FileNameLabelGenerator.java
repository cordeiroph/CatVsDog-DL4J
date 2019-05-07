package catvsdog;


import org.apache.commons.io.FilenameUtils;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;

import java.io.File;
import java.net.URI;

public class FileNameLabelGenerator implements PathLabelGenerator {

    public FileNameLabelGenerator() {
    }

    @Override
    public Writable getLabelForPath(String path) {
        String dirName = FilenameUtils.getName(new File(path).getName());
        String[] names = dirName.split(".");
        if (dirName.toLowerCase().contains("dog")){
            return new Text("0");
        }
        else{
            return new Text("1");
        }
    }

    @Override
    public Writable getLabelForPath(URI uri) {
        return getLabelForPath(new File(uri).toString());
    }

    @Override
    public boolean inferLabelClasses() {
        return true;
    }
}