/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 13:54:17 GMT 2023
 */

package org.jsoup.parser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Attributes;
import org.jsoup.nodes.Comment;
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
      htmlTreeBuilder0.parse("table", "table");
      Element element0 = htmlTreeBuilder0.insertStartTag("table");
      ParseErrorList parseErrorList0 = new ParseErrorList(492, 492);
      htmlTreeBuilder0.parseFragment("C", element0, "C", parseErrorList0);
      assertEquals(2, parseErrorList0.size());
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
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
  public void test02()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Element element0 = htmlTreeBuilder0.getHeadElement();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("avdress", "avdress");
      assertEquals(1, document0.childNodeSize());
      
      boolean boolean0 = htmlTreeBuilder0.processEndTag("avdress");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
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
  public void test05()  throws Throwable  {
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
  public void test06()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.generateImpliedEndTags();
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.insertMarkerToFormattingElements();
      htmlTreeBuilder0.clearFormattingElementsToLastMarker();
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      FormElement formElement0 = htmlTreeBuilder0.getFormElement();
      assertNull(formElement0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("avdress", "avdress");
      boolean boolean0 = htmlTreeBuilder0.processStartTag("select");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
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
  public void test11()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.setPendingTableCharacters((List<String>) null);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.inScope("2il1(");
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
      Document document0 = htmlTreeBuilder0.parse("<!doctype", "#root");
      assertEquals(2, document0.childNodeSize());
      assertEquals("#root", document0.location());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("avdres", "avdres");
      boolean boolean0 = htmlTreeBuilder0.isInActiveFormattingElements(document0);
      assertFalse(boolean0);
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      boolean boolean0 = htmlTreeBuilder0.isFragmentParsing();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
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
  public void test17()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.setFosterInserts(true);
      Document document0 = htmlTreeBuilder0.parse("table", "table");
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
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
  public void test19()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("m<?7Et ''V{", "m<?7Et ''V{");
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      List<Node> list0 = htmlTreeBuilder0.parseFragment("#data", (Element) null, "#data", parseErrorList0);
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      htmlTreeBuilder0.parse("th", "title", parseErrorList0);
      Element element0 = htmlTreeBuilder0.insertStartTag("title");
      List<Node> list0 = htmlTreeBuilder0.parseFragment("title", element0, "button", parseErrorList0);
      assertEquals(1, element0.siblingIndex());
      assertEquals(1, list0.size());
      assertEquals("title", element0.baseUri());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Document.createShell("iframe");
      Element element0 = document0.createElement("iframe");
      List<Node> list0 = htmlTreeBuilder0.parseFragment("iframe", element0, "5bE06RuZ<NhP,cb$R", (ParseErrorList) null);
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Document.createShell("plaintext");
      Element element0 = document0.createElement("script");
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      List<Node> list0 = htmlTreeBuilder0.parseFragment("script", element0, "Zs3^!$8", parseErrorList0);
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("table", "table");
      Element element0 = htmlTreeBuilder0.insertStartTag("noscript");
      ParseErrorList parseErrorList0 = htmlTreeBuilder0.errors;
      List<Node> list0 = htmlTreeBuilder0.parseFragment("noscript", element0, "noscript", parseErrorList0);
      assertEquals(1, list0.size());
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("org.jsoup.nodes.Element", "th");
      Element element0 = htmlTreeBuilder0.insertStartTag("plaintext");
      ParseErrorList parseErrorList0 = htmlTreeBuilder0.errors;
      List<Node> list0 = htmlTreeBuilder0.parseFragment("plaintext", element0, "plaintext", parseErrorList0);
      assertEquals(1, list0.size());
      assertEquals(1, element0.siblingIndex());
      assertEquals("th", element0.baseUri());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Tag tag0 = Tag.valueOf("op_tgrou%");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "7$z8oN$^4+:", attributes0);
      List<Node> list0 = htmlTreeBuilder0.parseFragment("7$z8oN$^4+:", formElement0, "7$z8oN$^4+:", (ParseErrorList) null);
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("table", "table");
      htmlTreeBuilder0.maybeSetBaseUri(document0);
      assertEquals(1, document0.childNodeSize());
      assertEquals("table", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Token.StartTag token_StartTag0 = new Token.StartTag();
      token_StartTag0.selfClosing = true;
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.insert(token_StartTag0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Must be false
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("org.jsoup.parser.OlTreBui8der", "org.jsoup.parser.OlTreBui8der");
      Token.StartTag token_StartTag0 = new Token.StartTag();
      token_StartTag0.selfClosing = true;
      token_StartTag0.appendTagName('a');
      Element element0 = htmlTreeBuilder0.insertEmpty(token_StartTag0);
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("avdre)s", "avdre)s");
      Token.StartTag token_StartTag0 = new Token.StartTag();
      token_StartTag0.selfClosing = true;
      token_StartTag0.appendTagName('D');
      Element element0 = htmlTreeBuilder0.insertEmpty(token_StartTag0);
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("table", "table");
      Token.StartTag token_StartTag0 = new Token.StartTag();
      Attributes attributes0 = new Attributes();
      Token.StartTag token_StartTag1 = token_StartTag0.nameAttr("table", attributes0);
      FormElement formElement0 = htmlTreeBuilder0.insertForm(token_StartTag1, true);
      assertEquals(1, formElement0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("avdres", "avdres");
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
  public void test33()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("style", "style");
      assertEquals(1, document0.childNodeSize());
      
      boolean boolean0 = htmlTreeBuilder0.inListItemScope("?U`HB");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
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
  public void test35()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("avdress", "avdress");
      boolean boolean0 = htmlTreeBuilder0.removeFromStack(document0);
      assertEquals(1, document0.childNodeSize());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("avdress", "avdress");
      Element element0 = htmlTreeBuilder0.currentElement();
      boolean boolean0 = htmlTreeBuilder0.removeFromStack(element0);
      assertTrue(boolean0);
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("avdress", "avdress");
      htmlTreeBuilder0.popStackToClose("avdress");
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("tPblH", "tPblH");
      htmlTreeBuilder0.insertStartTag("no2cript");
      boolean boolean0 = htmlTreeBuilder0.processEndTag("no2cript");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("avdress", "avdress");
      String[] stringArray0 = new String[2];
      stringArray0[0] = "avdress";
      stringArray0[1] = "avdress";
      htmlTreeBuilder0.popStackToClose(stringArray0);
      htmlTreeBuilder0.clearStackToTableRowContext();
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("avdress", "avdress");
      htmlTreeBuilder0.insertStartTag("noscript");
      String[] stringArray0 = new String[7];
      stringArray0[0] = "noscript";
      htmlTreeBuilder0.popStackToClose(stringArray0);
      assertEquals(7, stringArray0.length);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("", "");
      htmlTreeBuilder0.popStackToBefore("");
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      htmlTreeBuilder0.parse("body", "body", parseErrorList0);
      htmlTreeBuilder0.popStackToBefore("body");
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("table", "table");
      htmlTreeBuilder0.insertStartTag("table");
      htmlTreeBuilder0.clearStackToTableContext();
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("avdress", "avdress");
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
  public void test45()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("table", "table");
      Element element0 = htmlTreeBuilder0.insertStartTag("table");
      Comment comment0 = new Comment("table", "form");
      element0.replaceWith(comment0);
      htmlTreeBuilder0.insertStartTag("YF");
      htmlTreeBuilder0.insertInFosterParent(element0);
      assertEquals(1, element0.childNodeSize());
      assertEquals(2, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("h66IRW88NJ{B", "h66IRW88NJ{B");
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
  public void test47()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Token.StartTag token_StartTag0 = new Token.StartTag();
      Document document0 = htmlTreeBuilder0.parse("table", "table");
      token_StartTag0.appendTagName('P');
      Element element0 = htmlTreeBuilder0.insertEmpty(token_StartTag0);
      htmlTreeBuilder0.insert(token_StartTag0);
      htmlTreeBuilder0.insertOnStackAfter(element0, document0);
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("avdress", "avdress");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.replaceActiveFormattingElement(document0, document0);
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      ParseErrorList parseErrorList0 = ParseErrorList.tracking(2030);
      htmlTreeBuilder0.parse("[", "[", parseErrorList0);
      Element element0 = htmlTreeBuilder0.insertStartTag("select");
      List<Node> list0 = htmlTreeBuilder0.parseFragment("select", element0, "select", parseErrorList0);
      assertEquals(1, list0.size());
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("tF", "tF");
      htmlTreeBuilder0.insertStartTag("td");
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("tR", "tR");
      htmlTreeBuilder0.insertStartTag("tR");
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("tF", "tF");
      htmlTreeBuilder0.insertStartTag("tbody");
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("thead", "thead");
      htmlTreeBuilder0.insertStartTag("thead");
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("z<UN,|qX2m", "tfoot");
      htmlTreeBuilder0.insertStartTag("tfoot");
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("caption", "caption");
      htmlTreeBuilder0.insertStartTag("caption");
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("colgroup", "colgroup");
      htmlTreeBuilder0.insertStartTag("colgroup");
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("table", "head");
      htmlTreeBuilder0.insertStartTag("head");
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("avdres", "avdres");
      htmlTreeBuilder0.insertStartTag("frameset");
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("fallback", "html");
      htmlTreeBuilder0.insertStartTag("html");
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("table", "table");
      htmlTreeBuilder0.insertStartTag("noscript");
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("avdress", "avdress");
      htmlTreeBuilder0.popStackToClose("avdress");
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.inButtonScope("avdress");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Should not be reachable
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("body", "body");
      assertEquals(1, document0.childNodeSize());
      
      boolean boolean0 = htmlTreeBuilder0.inButtonScope("body");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("mavdres", "mavdres");
      assertEquals(1, document0.childNodeSize());
      
      boolean boolean0 = htmlTreeBuilder0.inTableScope("mavdres");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("avdress", "avdress");
      htmlTreeBuilder0.clearStackToTableRowContext();
      htmlTreeBuilder0.pop();
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.inSelectScope("avdress");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Should not be reachable
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("table", "table");
      Element element0 = htmlTreeBuilder0.insertStartTag("table");
      assertEquals(1, element0.siblingIndex());
      
      boolean boolean0 = htmlTreeBuilder0.inSelectScope("table");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      htmlTreeBuilder0.parse("table", "table", parseErrorList0);
      Element element0 = htmlTreeBuilder0.insertStartTag("option");
      assertEquals(1, element0.siblingIndex());
      
      boolean boolean0 = htmlTreeBuilder0.inSelectScope("table");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("J=i?M{r^}O#^5l6t3O", "J=i?M{r^}O#^5l6t3O");
      htmlTreeBuilder0.insertStartTag("optgroup");
      htmlTreeBuilder0.generateImpliedEndTags("B#rG");
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Element element0 = htmlTreeBuilder0.removeLastFormattingElement();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Document.createShell("org.jsoup.parser.XlTreBui8der");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.clearFormattingElementsToLastMarker();
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Document.createShell("J=i?M{r^}O#^5l6t3O");
      Document document1 = htmlTreeBuilder0.parse("TLq,o8Gg.W&Vwv*)?U", "J=i?M{r^}O#^5l6t3O");
      document1.dataset();
      htmlTreeBuilder0.pushActiveFormattingElements(document1);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("avdres", "avdres");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      ParseErrorList parseErrorList0 = ParseErrorList.tracking(786);
      Document document0 = htmlTreeBuilder0.parse("", "", parseErrorList0);
      Element element0 = htmlTreeBuilder0.insertStartTag("marquee");
      htmlTreeBuilder0.pushActiveFormattingElements(element0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.reconstructFormattingElements();
  }

  @Test(timeout = 4000)
  public void test73()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("[", "[", (ParseErrorList) null);
      Element element0 = htmlTreeBuilder0.insertStartTag("[");
      htmlTreeBuilder0.pushActiveFormattingElements(element0);
      htmlTreeBuilder0.reconstructFormattingElements();
  }

  @Test(timeout = 4000)
  public void test74()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      Document document0 = htmlTreeBuilder0.parse("J=i?M{r^}O#^5l6t3O", "J=i?M{r^}O#^5l6t3O");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      List<Node> list0 = htmlTreeBuilder0.parseFragment("J=i?M{r^}O#^5l6t3O", document0, "J=i?M{r^}O#^5l6t3O", parseErrorList0);
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test75()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("table", "table");
      htmlTreeBuilder0.insertMarkerToFormattingElements();
      ParseErrorList parseErrorList0 = htmlTreeBuilder0.errors;
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      List<Node> list0 = htmlTreeBuilder0.parseFragment("noscript", document0, "noscript", parseErrorList0);
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test76()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("avdress", "avdress");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.removeFromActiveFormattingElements(document0);
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test77()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("hr", "hr");
      htmlTreeBuilder0.insertMarkerToFormattingElements();
      htmlTreeBuilder0.removeFromActiveFormattingElements(document0);
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test78()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.insertMarkerToFormattingElements();
      Element element0 = htmlTreeBuilder0.getActiveFormattingElement("fr2%ka?=3hr");
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test79()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      ParseErrorList parseErrorList0 = ParseErrorList.tracking(786);
      Document document0 = htmlTreeBuilder0.parse("", "", parseErrorList0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      Element element0 = htmlTreeBuilder0.getActiveFormattingElement("fr2%ka?=3hr");
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test80()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      ParseErrorList parseErrorList0 = new ParseErrorList(2030, 2030);
      htmlTreeBuilder0.parse("[", "[", parseErrorList0);
      Element element0 = htmlTreeBuilder0.insertStartTag("select");
      htmlTreeBuilder0.pushActiveFormattingElements(element0);
      Element element1 = htmlTreeBuilder0.getActiveFormattingElement("select");
      assertEquals(1, element1.siblingIndex());
      assertNotNull(element1);
  }

  @Test(timeout = 4000)
  public void test81()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("table", "table");
      Element element0 = htmlTreeBuilder0.insertStartTag("table");
      htmlTreeBuilder0.insertInFosterParent(element0);
      assertEquals(1, element0.siblingIndex());
  }
}
