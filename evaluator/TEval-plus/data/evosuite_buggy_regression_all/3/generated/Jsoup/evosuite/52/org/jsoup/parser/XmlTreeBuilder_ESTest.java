/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:51:57 GMT 2023
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
      Token.Doctype token_Doctype0 = new Token.Doctype();
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
  public void test1()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      xmlTreeBuilder0.parse("org.jsoup.parser.XmlTreeBuilder$1", "org.jsoup.parser.XmlTreeBuilder$1");
      xmlTreeBuilder0.processStartTag("org.jsoup.parser.XmlTreeBuilder$1");
      boolean boolean0 = xmlTreeBuilder0.processEndTag("#document");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      xmlTreeBuilder0.parseFragment("Unexpected token type: ", "Unexpected token type: ", (ParseErrorList) null);
      Token.StartTag token_StartTag0 = new Token.StartTag();
      Attributes attributes0 = new Attributes();
      token_StartTag0.selfClosing = true;
      Token.StartTag token_StartTag1 = token_StartTag0.nameAttr("Bt9ELTeSJ5 78%C*:7", attributes0);
      Element element0 = xmlTreeBuilder0.insert(token_StartTag1);
      assertEquals("bt9eltesj5 78%c*:7", element0.nodeName());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      xmlTreeBuilder0.parseFragment("Unexpected token type: ", "Unexpected token type: ", (ParseErrorList) null);
      Token.StartTag token_StartTag0 = new Token.StartTag();
      Attributes attributes0 = new Attributes();
      token_StartTag0.selfClosing = true;
      token_StartTag0.nameAttr("font", attributes0);
      Element element0 = xmlTreeBuilder0.insert(token_StartTag0);
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      xmlTreeBuilder0.parse("<!J ", "<!J ");
      Token.Comment token_Comment0 = new Token.Comment();
      xmlTreeBuilder0.insert(token_Comment0);
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      xmlTreeBuilder0.parse("<!J ", "<!J ");
      Token.Comment token_Comment0 = new Token.Comment();
      token_Comment0.bogus = true;
      xmlTreeBuilder0.insert(token_Comment0);
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      Document document0 = xmlTreeBuilder0.parse("<!?J ", "<!?J ");
      assertEquals("#root", document0.tagName());
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      xmlTreeBuilder0.parseFragment("Unexpected token type: ", "Unexpected token type: ", (ParseErrorList) null);
      boolean boolean0 = xmlTreeBuilder0.processEndTag("8radio");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      xmlTreeBuilder0.parse("org.jsoup.parser.XmlTreeBuilder$1", "org.jsoup.parser.XmlTreeBuilder$1");
      boolean boolean0 = xmlTreeBuilder0.processEndTag("#document");
      assertTrue(boolean0);
  }
}