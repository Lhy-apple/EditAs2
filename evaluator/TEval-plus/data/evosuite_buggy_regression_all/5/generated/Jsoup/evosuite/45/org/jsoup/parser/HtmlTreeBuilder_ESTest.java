/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 23:15:12 GMT 2023
 */

package org.jsoup.parser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.ArrayList;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Attributes;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.FormElement;
import org.jsoup.nodes.Node;
import org.jsoup.parser.HtmlTreeBuilder;
import org.jsoup.parser.ParseErrorList;
import org.jsoup.parser.Tag;
import org.jsoup.parser.Token;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class HtmlTreeBuilder_ESTest extends HtmlTreeBuilder_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse(" K'uTV3~4Z3.6B'hucF", " K'uTV3~4Z3.6B'hucF");
      Element element0 = htmlTreeBuilder0.insertStartTag("plaintext");
      element0.prepend("plaintext");
      assertEquals(1, element0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("<html>\n <head />\n <body>\n  te\n </body>\n</html>", "<html>\n <head />\n <body>\n  te\n </body>\n</html>");
      htmlTreeBuilder0.clearStackToTableRowContext();
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.replaceOnStack((Element) null, (Element) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.HtmlTreeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Element element0 = htmlTreeBuilder0.getHeadElement();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("ivkpdress", "ivkpdress");
      Token.StartTag token_StartTag0 = new Token.StartTag();
      Attributes attributes0 = new Attributes();
      Token.StartTag token_StartTag1 = token_StartTag0.nameAttr("attributeSingleValueCharsSorted", attributes0);
      FormElement formElement0 = htmlTreeBuilder0.insertForm(token_StartTag1, true);
      assertEquals(1, formElement0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.clearStackToTableBodyContext();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.HtmlTreeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.inListItemScope("ivkpdress");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.HtmlTreeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.toString();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.TreeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("tfoot", "tfoot");
      Element element0 = htmlTreeBuilder0.insertStartTag("tfoot");
      element0.prepend("datalist");
      assertEquals(1, element0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.generateImpliedEndTags();
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.insertMarkerToFormattingElements();
      Document document0 = htmlTreeBuilder0.parse("2@", "2@");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      Document document1 = htmlTreeBuilder0.parse("2@", "2@");
      assertFalse(document1.equals((Object)document0));
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      FormElement formElement0 = htmlTreeBuilder0.getFormElement();
      assertNull(formElement0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.inButtonScope(" RFlp0");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.HtmlTreeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.state();
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.inScope(htmlTreeBuilder0.TagsSearchInScope);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.HtmlTreeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.setPendingTableCharacters((List<String>) null);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.getDocument();
      assertNull(document0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      String string0 = htmlTreeBuilder0.getBaseUri();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = new Document("tJ");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.parse("tJ", "`Wj9A1~@h");
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      ParseErrorList parseErrorList0 = ParseErrorList.tracking(2030);
      Document document0 = htmlTreeBuilder0.parse("[", "[", parseErrorList0);
      boolean boolean0 = htmlTreeBuilder0.isInActiveFormattingElements(document0);
      assertEquals(1, document0.childNodeSize());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("avdress", "avdress");
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.replaceActiveFormattingElement(document0, document0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Must be true
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.setFosterInserts(true);
      Document document0 = htmlTreeBuilder0.parse("attributeSingleValueCharsSorted", "attributeSingleValueCharsSorted");
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.push((Element) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.HtmlTreeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("<!--", "h1");
      assertEquals(2, document0.childNodeSize());
      assertEquals("h1", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      List<Node> list0 = htmlTreeBuilder0.parseFragment("Bv?M", (Element) null, "Bv?M", (ParseErrorList) null);
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Tag tag0 = Tag.valueOf("param");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "table", attributes0);
      formElement0.prepend("param");
      assertEquals(1, formElement0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("2@", "2@");
      Element element0 = htmlTreeBuilder0.insertStartTag("title");
      element0.prepend("title");
      assertEquals(1, element0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("'it", "'it");
      Element element0 = htmlTreeBuilder0.insertStartTag("style");
      element0.prepend("style");
      assertEquals(1, element0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("systemId", "0,o% cffYBr");
      Element element0 = htmlTreeBuilder0.insertStartTag("script");
      element0.prepend("script");
      assertEquals(1, element0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("r[CNt]_=)00l^:`?z", "r[CNt]_=)00l^:`?z");
      Element element0 = htmlTreeBuilder0.insertStartTag("noscript");
      element0.prepend("r[CNt]_=)00l^:`?z");
      assertEquals(1, element0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("ivkpdress", "ivkpdress");
      htmlTreeBuilder0.maybeSetBaseUri(document0);
      assertEquals(1, document0.childNodeSize());
      assertEquals("ivkpdress", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      ParseErrorList parseErrorList0 = ParseErrorList.tracking(1383);
      Document document0 = htmlTreeBuilder0.parse("org.jsoup.nodes.FormElement", "org.jsoup.nodes.FormElement", parseErrorList0);
      assertEquals(1, document0.childNodeSize());
      
      boolean boolean0 = htmlTreeBuilder0.processEndTag("org.jsoup.nodes.FormElement");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      ParseErrorList parseErrorList0 = ParseErrorList.tracking(2030);
      htmlTreeBuilder0.parse("[", "[", parseErrorList0);
      Token.StartTag token_StartTag0 = new Token.StartTag();
      Attributes attributes0 = new Attributes();
      Token.StartTag token_StartTag1 = token_StartTag0.nameAttr("select", attributes0);
      Element element0 = htmlTreeBuilder0.insertEmpty(token_StartTag1);
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("head", "head");
      Token.StartTag token_StartTag0 = new Token.StartTag();
      token_StartTag0.selfClosing = true;
      token_StartTag0.appendTagName('m');
      Element element0 = htmlTreeBuilder0.insertEmpty(token_StartTag0);
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("'it", "'it");
      Token.StartTag token_StartTag0 = new Token.StartTag();
      Attributes attributes0 = new Attributes();
      token_StartTag0.selfClosing = true;
      Token.StartTag token_StartTag1 = token_StartTag0.nameAttr("tr", attributes0);
      Element element0 = htmlTreeBuilder0.insert(token_StartTag1);
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("2@", "2@");
      htmlTreeBuilder0.insertStartTag("script");
      Token.Character token_Character0 = new Token.Character();
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.insert(token_Character0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Tag tag0 = Tag.valueOf("style");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "pw%@@8A9<-D", attributes0);
      htmlTreeBuilder0.pushActiveFormattingElements(formElement0);
      Document document0 = htmlTreeBuilder0.parse("style", "plaintext");
      assertEquals("plaintext", document0.location());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      ParseErrorList parseErrorList0 = ParseErrorList.tracking(2030);
      htmlTreeBuilder0.parse("[", "[", parseErrorList0);
      Token.StartTag token_StartTag0 = new Token.StartTag();
      Attributes attributes0 = new Attributes();
      Token.StartTag token_StartTag1 = token_StartTag0.nameAttr("select", attributes0);
      FormElement formElement0 = htmlTreeBuilder0.insertForm(token_StartTag1, false);
      assertEquals(1, formElement0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("table", "table");
      Element element0 = htmlTreeBuilder0.insertStartTag("table");
      element0.unwrap();
      htmlTreeBuilder0.insertInFosterParent(element0);
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("2p", "2p");
      boolean boolean0 = htmlTreeBuilder0.removeFromStack(document0);
      assertFalse(boolean0);
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("2p", "2p");
      Element element0 = document0.body();
      boolean boolean0 = htmlTreeBuilder0.removeFromStack(element0);
      assertEquals(1, element0.siblingIndex());
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse(">c", ">c");
      htmlTreeBuilder0.popStackToClose(">c");
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("html", "html");
      htmlTreeBuilder0.popStackToClose("html");
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("te", "te");
      String[] stringArray0 = new String[0];
      htmlTreeBuilder0.popStackToClose(stringArray0);
      assertEquals(0, stringArray0.length);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse(":@0Dx\"SB~@h\"5~->dOV", "body");
      String[] stringArray0 = new String[8];
      stringArray0[0] = "body";
      htmlTreeBuilder0.popStackToClose(stringArray0);
      assertEquals(8, stringArray0.length);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("thead", "thead");
      htmlTreeBuilder0.popStackToBefore("thead");
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("table", "table");
      htmlTreeBuilder0.insertStartTag("table");
      htmlTreeBuilder0.popStackToBefore("table");
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("T", "T");
      htmlTreeBuilder0.popStackToClose("T");
      htmlTreeBuilder0.clearStackToTableRowContext();
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("r[CYt]_=)00l^:`?z", "r[CYt]_=)00l^:`?z");
      htmlTreeBuilder0.insertStartTag("table");
      htmlTreeBuilder0.clearStackToTableContext();
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("vress", "vress");
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.aboveOnStack(document0);
        fail("Expecting exception: AssertionError");
      
      } catch(AssertionError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("avkpdress", "avkpdress");
      Element element0 = htmlTreeBuilder0.insertStartTag("avkpdress");
      Element element1 = htmlTreeBuilder0.aboveOnStack(element0);
      assertNotNull(element1);
      
      Element element2 = htmlTreeBuilder0.aboveOnStack(element1);
      assertEquals(2, element2.childNodeSize());
      assertTrue(element2.isBlock());
      assertNotSame(element2, element1);
      assertEquals(1, element1.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("h66IRW88NJ{B", "auYOuaqsn%VU");
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.insertOnStackAfter(document0, document0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Must be true
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("ivkpdress", "ivkpdress");
      Element element0 = htmlTreeBuilder0.insertStartTag("ivkpdress");
      htmlTreeBuilder0.insertOnStackAfter(element0, document0);
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("avdress", "avdress");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.replaceActiveFormattingElement(document0, document0);
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("tJ", "`Wj9A1~@h");
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("'it", "'it");
      Element element0 = htmlTreeBuilder0.insertStartTag("select");
      element0.prepend("'it");
      assertEquals(1, element0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("'it", "'it");
      htmlTreeBuilder0.insertStartTag("td");
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("[%s$=%s]", "[%s$=%s]");
      htmlTreeBuilder0.insertStartTag("tr");
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("tbody", "tbody");
      Element element0 = htmlTreeBuilder0.insertStartTag("tbody");
      element0.prepend("tbody");
      assertEquals(1, element0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("thead", "thead");
      htmlTreeBuilder0.insertStartTag("thead");
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("'i", "'i");
      htmlTreeBuilder0.insertStartTag("caption");
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("hgroup", "2?aa8IVeQj9bFYhtR");
      htmlTreeBuilder0.insertStartTag("colgroup");
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("table", "table");
      Element element0 = htmlTreeBuilder0.insertStartTag("table");
      element0.prepend("table");
      assertEquals(1, element0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("avkpdress", "avkpdress");
      Element element0 = document0.head();
      element0.prepend("avkpdress");
      assertEquals(1, element0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("frameset", "frameset");
      Element element0 = htmlTreeBuilder0.insertStartTag("frameset");
      Element element1 = element0.prepend("frameset");
      assertEquals(1, element1.siblingIndex());
      assertEquals(0, element1.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("html", "html");
      htmlTreeBuilder0.insertStartTag("html");
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      ArrayList<Element> arrayList0 = new ArrayList<Element>();
      htmlTreeBuilder0.stack = arrayList0;
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.inScope("81AE,fb4`.2GY*m]");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Should not be reachable
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("9", "9");
      assertEquals(1, document0.childNodeSize());
      
      boolean boolean0 = htmlTreeBuilder0.inTableScope("9");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      ParseErrorList parseErrorList0 = ParseErrorList.tracking(2030);
      Document document0 = htmlTreeBuilder0.parse("[", "[", parseErrorList0);
      assertEquals(1, document0.childNodeSize());
      
      boolean boolean0 = htmlTreeBuilder0.inScope("[", htmlTreeBuilder0.TagsSearchInScope);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("2@", "2@");
      htmlTreeBuilder0.popStackToClose("2@");
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.inSelectScope("2@");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Should not be reachable
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("table", "table");
      assertEquals(1, document0.childNodeSize());
      
      boolean boolean0 = htmlTreeBuilder0.inSelectScope("table");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("table", "table");
      Element element0 = htmlTreeBuilder0.insertStartTag("table");
      assertEquals(1, element0.siblingIndex());
      
      boolean boolean0 = htmlTreeBuilder0.inSelectScope("table");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("body", "body");
      htmlTreeBuilder0.generateImpliedEndTags("body");
  }

  @Test(timeout = 4000)
  public void test73()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("ivkpdress", "ivkpdress");
      htmlTreeBuilder0.generateImpliedEndTags("]]'@?F");
  }

  @Test(timeout = 4000)
  public void test74()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Element element0 = htmlTreeBuilder0.removeLastFormattingElement();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test75()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = new Document("'ir");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.clearFormattingElementsToLastMarker();
  }

  @Test(timeout = 4000)
  public void test76()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = new Document("tJ");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      Document document1 = htmlTreeBuilder0.parse("tJ", "`Wj9A1~@h");
      assertEquals(1, document1.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test77()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("pw%@@8A9<-D", "m=%}");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      Tag tag0 = Tag.valueOf("style");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "pw%@@8A9<-D", attributes0);
      htmlTreeBuilder0.pushActiveFormattingElements(formElement0);
      assertEquals(0, formElement0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test78()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("lyG", "lyG");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test79()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Document.createShell("'ir");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.parse("'ir", "'ir");
      htmlTreeBuilder0.reconstructFormattingElements();
  }

  @Test(timeout = 4000)
  public void test80()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("lyG", "lyG");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.parse("}d", "tJ");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.reconstructFormattingElements();
  }

  @Test(timeout = 4000)
  public void test81()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.insertMarkerToFormattingElements();
      htmlTreeBuilder0.clearFormattingElementsToLastMarker();
  }

  @Test(timeout = 4000)
  public void test82()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = new Document("vress");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.removeFromActiveFormattingElements(document0);
      assertEquals(0, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test83()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = new Document("vress");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      Document document1 = htmlTreeBuilder0.parse("vress", "vress");
      htmlTreeBuilder0.removeFromActiveFormattingElements(document1);
      assertEquals(1, document1.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test84()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = new Document("tJ");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      Document document1 = (Document)htmlTreeBuilder0.getActiveFormattingElement("#document");
      assertEquals(Document.QuirksMode.noQuirks, document1.quirksMode());
  }

  @Test(timeout = 4000)
  public void test85()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.insertMarkerToFormattingElements();
      Element element0 = htmlTreeBuilder0.getActiveFormattingElement("");
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test86()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("z91%", "z91%");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      Element element0 = htmlTreeBuilder0.getActiveFormattingElement("z91%");
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test87()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("table", "table");
      Element element0 = htmlTreeBuilder0.insertStartTag("table");
      htmlTreeBuilder0.insertInFosterParent(element0);
      assertEquals(1, element0.siblingIndex());
  }
}
