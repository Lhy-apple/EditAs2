/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:17:59 GMT 2023
 */

package org.jsoup.parser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.StringReader;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Document;
import org.jsoup.parser.Parser;
import org.jsoup.parser.Token;
import org.jsoup.parser.XmlTreeBuilder;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class XmlTreeBuilder_ESTest extends XmlTreeBuilder_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      Document document0 = Document.createShell("fo/.$");
      Parser parser0 = Parser.xmlParser();
      xmlTreeBuilder0.parseFragment("|{N*[wi/", document0, "fo/.$", parser0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      Token.Doctype token_Doctype0 = new Token.Doctype();
      // Undeclared exception!
      try { 
        xmlTreeBuilder0.process(token_Doctype0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.XmlTreeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      StringReader stringReader0 = new StringReader("progress");
      Document document0 = xmlTreeBuilder0.parse(stringReader0, "progress");
      assertEquals("progress", document0.location());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      Document document0 = xmlTreeBuilder0.parse("Jp]<bH1CGDdg$>t^3", "Jp]<bH1CGDdg$>t^3");
      assertEquals(2, document0.childNodeSize());
      
      boolean boolean0 = xmlTreeBuilder0.processEndTag("Jp]<bH1CGDdg$>t^3");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      Token.Comment token_Comment0 = new Token.Comment();
      // Undeclared exception!
      try { 
        xmlTreeBuilder0.process(token_Comment0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.TreeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      Token.Comment token_Comment0 = new Token.Comment();
      token_Comment0.bogus = true;
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      // Undeclared exception!
      try { 
        xmlTreeBuilder0.insert(token_Comment0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.TreeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      Token.CData token_CData0 = new Token.CData("s P");
      // Undeclared exception!
      try { 
        xmlTreeBuilder0.process(token_CData0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.TreeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      Document document0 = xmlTreeBuilder0.parse("Jp]<bH1CGDdg$>t^3", "Jp]<bH1CGDdg$>t^3");
      assertEquals(2, document0.childNodeSize());
      
      boolean boolean0 = xmlTreeBuilder0.processEndTag("#document");
      assertTrue(boolean0);
  }
}