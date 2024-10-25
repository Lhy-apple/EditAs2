/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:28:23 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.DocumentType;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DocumentType_ESTest extends DocumentType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      DocumentType documentType0 = new DocumentType("<!DOCTYPE", "", "", "`ycm+C:");
      StringBuilder stringBuilder0 = new StringBuilder((CharSequence) "");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      documentType0.outerHtmlHead(stringBuilder0, 610, document_OutputSettings0);
      assertEquals("<!DOCTYPE <!DOCTYPE>", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      DocumentType documentType0 = new DocumentType("`ycmLC:", "`ycmLC:", "`ycmLC:", "`ycmLC:");
      String string0 = documentType0.outerHtml();
      assertEquals("<!DOCTYPE `ycmLC: PUBLIC \"`ycmLC:\" \"`ycmLC:\">", string0);
      assertEquals("#doctype", documentType0.nodeName());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      StringBuilder stringBuilder0 = new StringBuilder((CharSequence) "Rcdata");
      DocumentType documentType0 = new DocumentType(" ", " ", "Rcdata", "");
      documentType0.outerHtmlHead(stringBuilder0, 598, (Document.OutputSettings) null);
      assertEquals("Rcdata<!DOCTYPE \"Rcdata\">", stringBuilder0.toString());
  }
}
