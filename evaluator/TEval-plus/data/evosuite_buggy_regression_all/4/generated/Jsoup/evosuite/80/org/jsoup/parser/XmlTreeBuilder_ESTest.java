/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 21:56:51 GMT 2023
 */

package org.jsoup.parser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.StringReader;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Attributes;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Node;
import org.jsoup.parser.ParseErrorList;
import org.jsoup.parser.ParseSettings;
import org.jsoup.parser.Token;
import org.jsoup.parser.Tokeniser;
import org.jsoup.parser.XmlTreeBuilder;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class XmlTreeBuilder_ESTest extends XmlTreeBuilder_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      xmlTreeBuilder0.parse("<?ZMJ!b", "<?ZMJ!b");
      Tokeniser tokeniser0 = xmlTreeBuilder0.tokeniser;
      Token.StartTag token_StartTag0 = tokeniser0.startPending;
      token_StartTag0.selfClosing = true;
      Attributes attributes0 = new Attributes();
      token_StartTag0.nameAttr("p", attributes0);
      xmlTreeBuilder0.insert(token_StartTag0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      ParseSettings parseSettings0 = xmlTreeBuilder0.defaultSettings();
      List<Node> list0 = xmlTreeBuilder0.parseFragment("#document", "#document", (ParseErrorList) null, parseSettings0);
      assertFalse(list0.isEmpty());
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
      StringReader stringReader0 = new StringReader("h2Qcy");
      Document document0 = xmlTreeBuilder0.parse(stringReader0, "h2Qcy");
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      xmlTreeBuilder0.parse("<?Z:MJ!b", "<?Z:MJ!b");
      Tokeniser tokeniser0 = xmlTreeBuilder0.tokeniser;
      Token.StartTag token_StartTag0 = tokeniser0.startPending;
      Attributes attributes0 = new Attributes();
      token_StartTag0.nameAttr("<?Z:MJ!b", attributes0);
      token_StartTag0.selfClosing = true;
      Element element0 = xmlTreeBuilder0.insert(token_StartTag0);
      assertTrue(element0.hasParent());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      xmlTreeBuilder0.parse("<?ZMJ!b", "<?ZMJ!b");
      Tokeniser tokeniser0 = xmlTreeBuilder0.tokeniser;
      Token.Comment token_Comment0 = tokeniser0.commentPending;
      xmlTreeBuilder0.insert(token_Comment0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      Document document0 = xmlTreeBuilder0.parse("<?", "<?");
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      Document document0 = xmlTreeBuilder0.parse("<!=ye(2Z#x", "<!=ye(2Z#x");
      assertEquals("<!=ye(2Z#x", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      Token.CData token_CData0 = new Token.CData("");
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
  public void test09()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      xmlTreeBuilder0.parse("<?ZMJ!b", "<?ZMJ!b");
      boolean boolean0 = xmlTreeBuilder0.processEndTag("org.jsou.seect.StructuralEvaluatr$Root");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      xmlTreeBuilder0.parse("<?ZMJ!b", "<?ZMJ!b");
      xmlTreeBuilder0.processStartTag("<?ZMJ!b");
      boolean boolean0 = xmlTreeBuilder0.processEndTag("<?ZMJ!b");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      Document document0 = xmlTreeBuilder0.parse("x%(", "#document");
      assertEquals("#document", document0.baseUri());
      
      xmlTreeBuilder0.processStartTag("x%(");
      boolean boolean0 = xmlTreeBuilder0.processEndTag("#document");
      assertTrue(boolean0);
  }
}