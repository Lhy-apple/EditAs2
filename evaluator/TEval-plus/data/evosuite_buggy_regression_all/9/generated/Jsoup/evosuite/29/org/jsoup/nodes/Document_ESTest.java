/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:05:26 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Entities;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Document_ESTest extends Document_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Document document0 = Document.createShell("TD[4u6^5X}4F");
      document0.prependText("TD[4u6^5X}4F");
      String string0 = document0.html();
      assertEquals("TD[4u6^5X}4F\n<html>\n <head></head>\n <body></body>\n</html>", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      document_OutputSettings0.charset();
      assertEquals(1, document_OutputSettings0.indentAmount());
      assertTrue(document_OutputSettings0.prettyPrint());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.prettyPrint(true);
      assertEquals(1, document_OutputSettings1.indentAmount());
      assertTrue(document_OutputSettings0.prettyPrint());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Entities.EscapeMode entities_EscapeMode0 = Entities.EscapeMode.base;
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.escapeMode(entities_EscapeMode0);
      assertEquals(1, document_OutputSettings1.indentAmount());
      assertTrue(document_OutputSettings1.prettyPrint());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Document document0 = Document.createShell("P[R9iQ2t+F#v");
      document0.html("P[R9iQ2t+F#v");
      Document document1 = document0.normalise();
      assertEquals("#document", document1.nodeName());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Document document0 = Document.createShell("@9]");
      Element element0 = document0.createElement("@9]");
      assertEquals("@9]", element0.tagName());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Document document0 = new Document("#et");
      String string0 = document0.outerHtml();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Document document0 = new Document("kC%\"!");
      // Undeclared exception!
      try { 
        document0.outputSettings((Document.OutputSettings) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Document document0 = Document.createShell("d{5H0<A(:3R]Kt'9");
      String string0 = document0.title();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Document document0 = Document.createShell("#text");
      document0.title("#text");
      document0.title();
      assertEquals("#document", document0.nodeName());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Document document0 = Document.createShell("");
      document0.title("b \u0001cysNn5");
      document0.title("");
      assertEquals("#document", document0.nodeName());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Document document0 = new Document("?y)k6!gof");
      document0.appendText((String) null);
      Document document1 = document0.normalise();
      assertEquals("#document", document1.nodeName());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Document document0 = Document.createShell("body");
      Document document1 = document0.clone();
      document0.text("fVb:|l9_GaJ+@]MV");
      document1.appendChild(document0);
      document1.normalise();
      assertEquals(1, document0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Document document0 = new Document("#text");
      document0.prependElement("body");
      Document document1 = document0.normalise();
      assertEquals("#document", document1.nodeName());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      // Undeclared exception!
      try { 
        document_OutputSettings0.indentAmount((-1));
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Must be true
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      assertEquals(1, document_OutputSettings0.indentAmount());
      
      document_OutputSettings0.indentAmount(0);
      assertEquals(0, document_OutputSettings0.indentAmount());
  }
}