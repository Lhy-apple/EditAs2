/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:09:40 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import java.io.OutputStreamWriter;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.evosuite.runtime.mock.java.io.MockFileOutputStream;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.DocumentType;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DocumentType_ESTest extends DocumentType_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      DocumentType documentType0 = new DocumentType("", "systemId", "", "systemId");
      MockFileOutputStream mockFileOutputStream0 = new MockFileOutputStream("systemId");
      OutputStreamWriter outputStreamWriter0 = new OutputStreamWriter(mockFileOutputStream0);
      documentType0.html(outputStreamWriter0);
      assertEquals("#doctype", documentType0.nodeName());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      DocumentType documentType0 = new DocumentType("it p`<d", "it p`<d", "it p`<d", "it p`<d");
      StringBuilder stringBuilder0 = new StringBuilder("it p`<d");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings.Syntax document_OutputSettings_Syntax0 = Document.OutputSettings.Syntax.xml;
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.syntax(document_OutputSettings_Syntax0);
      documentType0.outerHtmlHead(stringBuilder0, 34, document_OutputSettings1);
      assertEquals("it p`<d<!DOCTYPE it p`<d PUBLIC \"it p`<d\" \"it p`<d\">", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      DocumentType documentType0 = new DocumentType("", "", "a", "a");
      String string0 = documentType0.outerHtml();
      assertEquals("#doctype", documentType0.nodeName());
      assertEquals("<!DOCTYPE \"a\">", string0);
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      DocumentType documentType0 = new DocumentType("a", "", "", "\n");
      String string0 = documentType0.outerHtml();
      assertEquals("#doctype", documentType0.nodeName());
      assertEquals("<!doctype a>", string0);
  }
}