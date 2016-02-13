find_program(asciidoc_cmd NAMES asciidoc)
if(EXISTS ${asciidoc_cmd}) 

add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/documentation.html
		    #COMMAND ${asciidoc_cmd} --filter install ${CMAKE_CURRENT_LIST_DIR}/doc/plantuml.zip
		    COMMAND cp -rf ${CMAKE_CURRENT_SOURCE_DIR}/structural/images ${CMAKE_CURRENT_BINARY_DIR}
		    COMMAND ${asciidoc_cmd} --verbose -dbook -o ${CMAKE_CURRENT_BINARY_DIR}/documentation.html ${CMAKE_CURRENT_LIST_DIR}/structural/documentation.txt
		    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/structural/*)
		    
add_custom_target(documentation ALL
		  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/documentation.html)
		    

endif()