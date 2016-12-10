
def xml(filename, size, class_label, bndbox):
    string = """<annotation>
    	<folder>VOC2012</folder>
    	<filename>{:s}</filename>
    	<source>
    		<database>The VOC2007 Database</database>
    		<annotation>PASCAL VOC2007</annotation>
    		<image>flickr</image>
    	</source>
    	<size>
    		<width>{:d}</width>
    		<height>{:d}</height>
    		<depth>{:d}</depth>
    	</size>
    	<segmented>0</segmented>
    	<object>
    		<name>{:s}</name>
    		<pose>Unspecified</pose>
    		<truncated>0</truncated>
    		<difficult>0</difficult>
    		<bndbox>
    			<xmin>{:d}</xmin>
    			<ymin>{:d}</ymin>
    			<xmax>{:d}</xmax>
    			<ymax>{:d}</ymax>
    		</bndbox>

    	</object>
    </annotation>""".format(
        filename, size[0], size[1], size[2], class_label,
        bndbox[0], bndbox[1], bndbox[2], bndbox[3])
    return string


def create_file(filename, size, class_label, bndbox):
    xml_file = filename.split('.')[0]+'.xml'
    xmlS = xml(filename, size, class_label, bndbox)
    with open(xml_file, 'w') as f:
        f.write(xmlS)
        f.flush()
        f.close()
