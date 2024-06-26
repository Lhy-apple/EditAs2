/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 03:36:41 GMT 2023
 */

package org.jsoup.parser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.PipedReader;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Attributes;
import org.jsoup.nodes.CDataNode;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Node;
import org.jsoup.parser.ParseErrorList;
import org.jsoup.parser.ParseSettings;
import org.jsoup.parser.Token;
import org.jsoup.parser.XmlTreeBuilder;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class XmlTreeBuilder_ESTest extends XmlTreeBuilder_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      // Undeclared exception!
      try { 
        xmlTreeBuilder0.parse("I`%}<?P", "I`%}<?P");
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 0, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      ParseSettings parseSettings0 = ParseSettings.htmlDefault;
      List<Node> list0 = xmlTreeBuilder0.parseFragment("|T.xsG<UC>ZSw", "|T.xsG<UC>ZSw", (ParseErrorList) null, parseSettings0);
      assertEquals(2, list0.size());
      
      boolean boolean0 = xmlTreeBuilder0.processEndTag("#document");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
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
  public void test03()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      PipedReader pipedReader0 = new PipedReader();
      // Undeclared exception!
      try { 
        xmlTreeBuilder0.parse(pipedReader0, "=|B^5lj,.xWES]");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Must be true
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      xmlTreeBuilder0.parse(" %s>", " %s>");
      Token.StartTag token_StartTag0 = new Token.StartTag();
      CDataNode cDataNode0 = new CDataNode("<!---->");
      Attributes attributes0 = cDataNode0.attributes();
      token_StartTag0.selfClosing = true;
      Token.StartTag token_StartTag1 = token_StartTag0.nameAttr(" %s>", attributes0);
      Element element0 = xmlTreeBuilder0.insert(token_StartTag1);
      assertEquals("%s>", element0.tagName());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      xmlTreeBuilder0.parse(" %s>", " %s>");
      Token.StartTag token_StartTag0 = new Token.StartTag();
      token_StartTag0.selfClosing = true;
      token_StartTag0.tagName = "h5";
      Element element0 = xmlTreeBuilder0.insert(token_StartTag0);
      assertEquals("h5", element0.nodeName());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      Token.Comment token_Comment0 = new Token.Comment();
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
  public void test07()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      Document document0 = xmlTreeBuilder0.parse("};P%}<?", "};P%}<?");
      assertEquals(2, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      Document document0 = xmlTreeBuilder0.parse("<!OCTYPE", "<!OCTYPE");
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      Token.CData token_CData0 = new Token.CData("l|8q");
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
  public void test10()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      xmlTreeBuilder0.parse(" %s>", " %s>");
      boolean boolean0 = xmlTreeBuilder0.processEndTag("<!---->");
      assertTrue(boolean0);
  }
}
