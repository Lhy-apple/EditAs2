/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:51:42 GMT 2023
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
      DocumentType documentType0 = new DocumentType("command", "command", "command", "name");
      StringBuilder stringBuilder0 = new StringBuilder();
      Document.OutputSettings document_OutputSettings0 = documentType0.getOutputSettings();
      documentType0.outerHtmlTail(stringBuilder0, 38, document_OutputSettings0);
      assertEquals("", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      DocumentType documentType0 = new DocumentType("command", "command", "command", "name");
      String string0 = documentType0.nodeName();
      assertEquals("#doctype", string0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      DocumentType documentType0 = new DocumentType(" ", "5GmFa4FW'-~B", " ", "1");
      StringBuilder stringBuilder0 = new StringBuilder();
      Document.OutputSettings document_OutputSettings0 = documentType0.getOutputSettings();
      documentType0.outerHtmlHead(stringBuilder0, 2139, document_OutputSettings0);
      assertEquals("<!DOCTYPE PUBLIC \"5GmFa4FW'-~B\">", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      DocumentType documentType0 = new DocumentType("command", "command", "command", "name");
      StringBuilder stringBuilder0 = new StringBuilder();
      Document.OutputSettings document_OutputSettings0 = documentType0.getOutputSettings();
      documentType0.outerHtmlHead(stringBuilder0, 38, document_OutputSettings0);
      assertEquals("<!DOCTYPE command PUBLIC \"command\" \"command\">", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      DocumentType documentType0 = new DocumentType("command", "", ";OhGs-,88Ybt", ";OhGs-,88Ybt");
      StringBuilder stringBuilder0 = new StringBuilder();
      Document.OutputSettings document_OutputSettings0 = documentType0.getOutputSettings();
      documentType0.outerHtmlHead(stringBuilder0, 521, document_OutputSettings0);
      assertEquals("<!DOCTYPE command \";OhGs-,88Ybt\">", stringBuilder0.toString());
  }
}