/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:31:54 GMT 2023
 */

package org.jsoup.parser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Attributes;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.parser.ParseErrorList;
import org.jsoup.parser.Token;
import org.jsoup.parser.XmlTreeBuilder;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class XmlTreeBuilder_ESTest extends XmlTreeBuilder_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      xmlTreeBuilder0.parseFragment("Unexpected token type: ", "Unexpected token type: ", (ParseErrorList) null);
      Token.StartTag token_StartTag0 = new Token.StartTag();
      Attributes attributes0 = new Attributes();
      token_StartTag0.selfClosing = true;
      Token.StartTag token_StartTag1 = token_StartTag0.nameAttr("Bt9ELTeSJ5 78%C*:7", attributes0);
      Element element0 = xmlTreeBuilder0.insert(token_StartTag1);
      assertEquals("Unexpected token type:", element0.baseUri());
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      Token.Doctype token_Doctype0 = new Token.Doctype();
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      // Undeclared exception!
      try { 
        xmlTreeBuilder0.process(token_Doctype0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      Document document0 = xmlTreeBuilder0.parse("9!Fq(Y<?oA;Vo236?G", "2L! ");
      assertEquals("2L! ", document0.location());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      xmlTreeBuilder0.parse("cD7+|11", "cD7+|11");
      xmlTreeBuilder0.processStartTag("cD7+|11");
      boolean boolean0 = xmlTreeBuilder0.processEndTag("#document");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      xmlTreeBuilder0.parseFragment("Unexpected token type: ", "Unexpected token type: ", (ParseErrorList) null);
      Token.StartTag token_StartTag0 = new Token.StartTag();
      Attributes attributes0 = new Attributes();
      token_StartTag0.selfClosing = true;
      token_StartTag0.nameAttr("font", attributes0);
      Element element0 = xmlTreeBuilder0.insert(token_StartTag0);
      assertEquals("font", element0.tagName());
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      xmlTreeBuilder0.parse("<!J ", "<!J ");
      Token.Comment token_Comment0 = new Token.Comment();
      xmlTreeBuilder0.insert(token_Comment0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      Document document0 = xmlTreeBuilder0.parse("<! ", "<! ");
      assertFalse(document0.updateMetaCharsetElement());
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      Document document0 = xmlTreeBuilder0.parse("<!! ", "org.soup.prser.XmlTreeBuilder$1");
      assertEquals("org.soup.prser.XmlTreeBuilder$1", document0.location());
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      xmlTreeBuilder0.initialiseParse("radio", "radio", (ParseErrorList) null);
      boolean boolean0 = xmlTreeBuilder0.processEndTag("?");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test9()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      xmlTreeBuilder0.parse("cD7+|11", "cD7+|11");
      boolean boolean0 = xmlTreeBuilder0.processEndTag("#document");
      assertTrue(boolean0);
  }
}
