/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 13:48:25 GMT 2023
 */

package org.jsoup.parser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.helper.DescendableLinkedList;
import org.jsoup.nodes.Attributes;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.FormElement;
import org.jsoup.nodes.Node;
import org.jsoup.parser.HtmlTreeBuilder;
import org.jsoup.parser.HtmlTreeBuilderState;
import org.jsoup.parser.ParseErrorList;
import org.jsoup.parser.Tag;
import org.jsoup.parser.Token;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class HtmlTreeBuilder_ESTest extends HtmlTreeBuilder_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("hA~%r2-31``Y;", "hA~%r2-31``Y;");
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
      DescendableLinkedList<Element> descendableLinkedList0 = htmlTreeBuilder0.getStack();
      assertNull(descendableLinkedList0);
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
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.toString();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("tbody", "tbody");
      Token.StartTag token_StartTag0 = new Token.StartTag("tbody");
      Element element0 = htmlTreeBuilder0.insertEmpty(token_StartTag0);
      ParseErrorList parseErrorList0 = htmlTreeBuilder0.errors;
      assertNotNull(parseErrorList0);
      
      htmlTreeBuilder0.parseFragment("tbody", element0, "", parseErrorList0);
      assertEquals(1, element0.siblingIndex());
      assertEquals(0, parseErrorList0.size());
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
      htmlTreeBuilder0.replaceActiveFormattingElement((Element) null, (Element) null);
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
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.clearStackToTableRowContext();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.HtmlTreeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("org.jsoup.parser.HtmlTreeBuilder", "select", (ParseErrorList) null);
      Token.StartTag token_StartTag0 = new Token.StartTag("select");
      HtmlTreeBuilderState htmlTreeBuilderState0 = HtmlTreeBuilderState.InCaption;
      boolean boolean0 = htmlTreeBuilder0.process(token_StartTag0, htmlTreeBuilderState0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.inScope((String[]) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.HtmlTreeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      boolean boolean0 = htmlTreeBuilder0.framesetOk();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      List<Token.Character> list0 = htmlTreeBuilder0.getPendingTableCharacters();
      htmlTreeBuilder0.setPendingTableCharacters(list0);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.getDocument();
      assertNull(document0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      String string0 = htmlTreeBuilder0.getBaseUri();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      boolean boolean0 = htmlTreeBuilder0.isInActiveFormattingElements((Element) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      boolean boolean0 = htmlTreeBuilder0.isFragmentParsing();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.replaceActiveFormattingElement((Element) null, (Element) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Must be true
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.setFosterInserts(true);
      Document document0 = htmlTreeBuilder0.parse("popping html!", "popping html!");
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
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
  public void test21()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.isSpecial((Element) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.HtmlTreeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Document document0 = Document.createShell("<!");
      document0.append("<!");
      assertEquals(2, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.inTableScope("Fb=tt)h");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.HtmlTreeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.clearStackToTableContext();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.HtmlTreeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      List<Node> list0 = htmlTreeBuilder0.parseFragment("base", (Element) null, "base", (ParseErrorList) null);
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("BU", "Unexpected token [%s] when in state [%s]");
      Element element0 = document0.createElement("plaintext");
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      List<Node> list0 = htmlTreeBuilder0.parseFragment("hgroup", element0, "org.jsoup.select.Evaluator$AttributeWithValueEnding", parseErrorList0);
      assertEquals(1, list0.size());
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("~ka#nc$yp>h7x~**t", "LFBl2a7_E=fh26&JN");
      Token.StartTag token_StartTag0 = new Token.StartTag("title");
      Element element0 = htmlTreeBuilder0.insertEmpty(token_StartTag0);
      ParseErrorList parseErrorList0 = htmlTreeBuilder0.errors;
      List<Node> list0 = htmlTreeBuilder0.parseFragment("LFBl2a7_E=fh26&JN", element0, "~ka#nc$yp>h7x~**t", parseErrorList0);
      assertEquals(1, element0.siblingIndex());
      assertEquals(1, list0.size());
      assertEquals("LFBl2a7_E=fh26&JN", element0.baseUri());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("(FR*rn", "(FR*rn");
      Token.StartTag token_StartTag0 = new Token.StartTag("style");
      Element element0 = htmlTreeBuilder0.insertEmpty(token_StartTag0);
      ParseErrorList parseErrorList0 = htmlTreeBuilder0.errors;
      List<Node> list0 = htmlTreeBuilder0.parseFragment("style", element0, "style", parseErrorList0);
      assertEquals(1, list0.size());
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Tag tag0 = Tag.valueOf("script");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "script", attributes0);
      List<Node> list0 = htmlTreeBuilder0.parseFragment("", formElement0, "script", (ParseErrorList) null);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Attributes attributes0 = new Attributes();
      Tag tag0 = Tag.valueOf("noscript");
      FormElement formElement0 = new FormElement(tag0, "meta", attributes0);
      List<Node> list0 = htmlTreeBuilder0.parseFragment("th", formElement0, "ul", (ParseErrorList) null);
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("(FR*rn", "(FR*rn");
      htmlTreeBuilder0.maybeSetBaseUri(document0);
      assertEquals(1, document0.childNodeSize());
      assertEquals("(FR*rn", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      ParseErrorList parseErrorList0 = ParseErrorList.tracking(25);
      htmlTreeBuilder0.parse("*|~Mk+Ng,L|+", ",}Fff@M]x3vDs{O", parseErrorList0);
      HtmlTreeBuilderState htmlTreeBuilderState0 = HtmlTreeBuilderState.InCaption;
      Token.Doctype token_Doctype0 = new Token.Doctype();
      boolean boolean0 = htmlTreeBuilder0.process(token_Doctype0, htmlTreeBuilderState0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse(";G[^+YGDea iffj?$", ";G[^+YGDea iffj?$");
      Token.StartTag token_StartTag0 = new Token.StartTag("figure");
      token_StartTag0.selfClosing = true;
      Element element0 = htmlTreeBuilder0.insert(token_StartTag0);
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("L[OVCEJ.b<&<]", "L[OVCEJ.b<&<]");
      Token.StartTag token_StartTag0 = new Token.StartTag("L[OVCEJ.b<&<]");
      token_StartTag0.selfClosing = true;
      Element element0 = htmlTreeBuilder0.insert(token_StartTag0);
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse(";G[^+YGDea iffj?$", ";G[^+YGDea iffj?$");
      Token.StartTag token_StartTag0 = new Token.StartTag("figure");
      token_StartTag0.selfClosing = true;
      token_StartTag0.name("h4");
      Element element0 = htmlTreeBuilder0.insert(token_StartTag0);
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("", "", (ParseErrorList) null);
      Token.StartTag token_StartTag0 = new Token.StartTag("select");
      FormElement formElement0 = htmlTreeBuilder0.insertForm(token_StartTag0, false);
      assertEquals(0, formElement0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("td", "td");
      htmlTreeBuilder0.insert("td");
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.pop();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // pop td not in cell
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("td", "td");
      htmlTreeBuilder0.insert("td");
      htmlTreeBuilder0.resetInsertionMode();
      Element element0 = htmlTreeBuilder0.pop();
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("InCell", "InCell");
      htmlTreeBuilder0.clearStackToTableBodyContext();
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.pop();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // popping html!
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("Rdd", "Rdd");
      Token.StartTag token_StartTag0 = new Token.StartTag("table");
      Element element0 = htmlTreeBuilder0.insert(token_StartTag0);
      htmlTreeBuilder0.insertInFosterParent(element0);
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("m`tOb3r/)P", "AfterBody", (ParseErrorList) null);
      boolean boolean0 = htmlTreeBuilder0.removeFromStack(document0);
      assertEquals("AfterBody", document0.baseUri());
      assertFalse(boolean0);
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("%QOb?r/CO)P", "dnilefjho=$7n.yze|'", (ParseErrorList) null);
      Element element0 = document0.body();
      boolean boolean0 = htmlTreeBuilder0.removeFromStack(element0);
      assertTrue(boolean0);
      assertEquals(1, element0.siblingIndex());
      assertEquals("dnilefjho=$7n.yze|'", element0.baseUri());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("J", "J");
      htmlTreeBuilder0.popStackToClose("J");
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("m`tOb3r/)P", "m`tOb3r/)P", (ParseErrorList) null);
      htmlTreeBuilder0.popStackToClose("body");
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("h3", "h3", (ParseErrorList) null);
      String[] stringArray0 = new String[0];
      htmlTreeBuilder0.popStackToClose(stringArray0);
      assertEquals(0, stringArray0.length);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("h3", "h3", (ParseErrorList) null);
      Token.StartTag token_StartTag0 = new Token.StartTag("h3");
      HtmlTreeBuilderState htmlTreeBuilderState0 = HtmlTreeBuilderState.InCaption;
      htmlTreeBuilder0.process(token_StartTag0, htmlTreeBuilderState0);
      String[] stringArray0 = new String[5];
      stringArray0[0] = "<0(*\"J*eT[|3dC7";
      stringArray0[1] = "<0(*\"J*eT[|3dC7";
      stringArray0[2] = "<0(*\"J*eT[|3dC7";
      stringArray0[3] = "h3";
      htmlTreeBuilder0.popStackToClose(stringArray0);
      assertEquals(5, stringArray0.length);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("m`tOb30r-)Q", "m`tOb30r-)Q");
      htmlTreeBuilder0.popStackToBefore("m`tOb30r-)Q");
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.inSelectScope("m`tOb30r-)Q");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Should not be reachable
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("m`tOb3r/)P", "AfterBody", (ParseErrorList) null);
      htmlTreeBuilder0.popStackToBefore("body");
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.initialiseParse("", "", (ParseErrorList) null);
      htmlTreeBuilder0.clearStackToTableBodyContext();
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("G^6KmDU}|h1HT0", "G^6KmDU}|h1HT0");
      Element element0 = htmlTreeBuilder0.insert("G^6KmDU}|h1HT0");
      Element element1 = htmlTreeBuilder0.aboveOnStack(element0);
      assertNotNull(element1);
      
      Element element2 = htmlTreeBuilder0.aboveOnStack(element1);
      assertEquals(2, element2.childNodeSize());
      assertEquals("html", element2.nodeName());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("(", "(");
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
      Document document0 = htmlTreeBuilder0.parse("*", "*");
      Element element0 = htmlTreeBuilder0.currentElement();
      htmlTreeBuilder0.insertOnStackAfter(element0, document0);
      assertEquals(1, element0.siblingIndex());
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("select", "select");
      Token.StartTag token_StartTag0 = new Token.StartTag("select");
      Element element0 = htmlTreeBuilder0.insertEmpty(token_StartTag0);
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      htmlTreeBuilder0.parseFragment("tobr/)", element0, "select", parseErrorList0);
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("tr", "tr");
      Token.StartTag token_StartTag0 = new Token.StartTag("tr");
      htmlTreeBuilder0.insert(token_StartTag0);
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Tag tag0 = Tag.valueOf("thead");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "f\f", attributes0);
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      htmlTreeBuilder0.parseFragment("Fb=ttoh", formElement0, "#C0%O~\"mSv+hb", parseErrorList0);
      assertTrue(parseErrorList0.isEmpty());
      assertEquals(0, parseErrorList0.size());
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Tag tag0 = Tag.valueOf("tfoot");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "tfoot", attributes0);
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      htmlTreeBuilder0.parseFragment("tfoot", formElement0, "tfoot", parseErrorList0);
      assertEquals(0, parseErrorList0.size());
      assertTrue(parseErrorList0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("caption", "caption");
      Token.StartTag token_StartTag0 = new Token.StartTag("caption");
      Element element0 = htmlTreeBuilder0.insertEmpty(token_StartTag0);
      ParseErrorList parseErrorList0 = htmlTreeBuilder0.errors;
      htmlTreeBuilder0.parseFragment(">\"9`<", element0, "caption", parseErrorList0);
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("colgroup", "colgroup");
      Token.StartTag token_StartTag0 = new Token.StartTag("colgroup");
      Element element0 = htmlTreeBuilder0.insertEmpty(token_StartTag0);
      ParseErrorList parseErrorList0 = htmlTreeBuilder0.errors;
      List<Node> list0 = htmlTreeBuilder0.parseFragment("colgroup", element0, "colgroup", parseErrorList0);
      assertEquals(0, list0.size());
      assertTrue(parseErrorList0.isEmpty());
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("*Aj^,V*b~iWpCS", "*Aj^,V*b~iWpCS");
      Token.StartTag token_StartTag0 = new Token.StartTag("table");
      Element element0 = htmlTreeBuilder0.insertEmpty(token_StartTag0);
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      htmlTreeBuilder0.parseFragment("embed", element0, "embed", parseErrorList0);
      assertTrue(parseErrorList0.isEmpty());
      assertEquals(1, element0.siblingIndex());
      assertEquals(0, parseErrorList0.size());
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("head", "head");
      htmlTreeBuilder0.insert("head");
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("frameset", "frameset");
      htmlTreeBuilder0.insert("frameset");
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("hA~%r2-31``Y;", "hA~%r2-31``Y;");
      Element element0 = document0.body();
      element0.wrap("hA~%r2-31``Y;");
      assertEquals(0, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("1^nx*5wzf;5lm/h6q", "1^nx*5wzf;5lm/h6q");
      htmlTreeBuilder0.insert("1^nx*5wzf;5lm/h6q");
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.initialiseParse("", "", (ParseErrorList) null);
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.inListItemScope("c9xHh33 +XQoZ%'");
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
      htmlTreeBuilder0.parse("td", "td");
      Element element0 = htmlTreeBuilder0.insert("td");
      assertEquals(1, element0.siblingIndex());
      
      boolean boolean0 = htmlTreeBuilder0.inListItemScope("td");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("Fb=tt)h", "Fb=tt)h");
      assertEquals(1, document0.childNodeSize());
      
      boolean boolean0 = htmlTreeBuilder0.inScope("Fb=tt)h");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("sk*$i*#Uo'(", "sk*$i*#Uo'(", (ParseErrorList) null);
      Token.StartTag token_StartTag0 = new Token.StartTag("tr");
      Element element0 = htmlTreeBuilder0.insert(token_StartTag0);
      assertEquals(1, element0.siblingIndex());
      
      boolean boolean0 = htmlTreeBuilder0.inSelectScope("tr");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("optgroup", "center", (ParseErrorList) null);
      Token.StartTag token_StartTag0 = new Token.StartTag("optgroup");
      Element element0 = htmlTreeBuilder0.insert(token_StartTag0);
      assertEquals(1, element0.siblingIndex());
      assertEquals("center", element0.baseUri());
      
      boolean boolean0 = htmlTreeBuilder0.inSelectScope("center");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("%s", "%s");
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      htmlTreeBuilder0.parseFragment("bdn*FVur~i$PAggy", document0, ")m@Ppkc#px6w", parseErrorList0);
      htmlTreeBuilder0.generateImpliedEndTags("html");
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("Fb=ttDd", "Fb=ttDd");
      Attributes attributes0 = new Attributes();
      Token.StartTag token_StartTag0 = new Token.StartTag("dt", attributes0);
      htmlTreeBuilder0.insertForm(token_StartTag0, true);
      htmlTreeBuilder0.generateImpliedEndTags("Fb=ttDd");
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Document.createShell("thead");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      assertEquals("#root", document0.tagName());
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("%QOb?r/CO)P", "%QOb?r/CO)P");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      Element element0 = htmlTreeBuilder0.pop();
      htmlTreeBuilder0.pushActiveFormattingElements(element0);
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test73()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.insertMarkerToFormattingElements();
      Document document0 = htmlTreeBuilder0.parse("h3", "h3", (ParseErrorList) null);
      htmlTreeBuilder0.removeFromActiveFormattingElements(document0);
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test74()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = new Document("+za/~-]{v@]k74");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.parse("+za/~-]{v@]k74", "+za/~-]{v@]k74");
      htmlTreeBuilder0.reconstructFormattingElements();
  }

  @Test(timeout = 4000)
  public void test75()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.insertMarkerToFormattingElements();
      Document document0 = Document.createShell("*|~Mk+Ng,L|+");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      Document document1 = htmlTreeBuilder0.parse("*|~Mk+Ng,L|+", "*|~Mk+Ng,L|+", (ParseErrorList) null);
      assertEquals(1, document1.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test76()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse(";G[^+YGDea iffj?$", ";G[^+YGDea iffj?$");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.parse("EOF", "u|} fe2p^4e#");
      htmlTreeBuilder0.pop();
      htmlTreeBuilder0.reconstructFormattingElements();
  }

  @Test(timeout = 4000)
  public void test77()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("m`tOb30r-)Q", "m`tOb30r-)Q");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.clearFormattingElementsToLastMarker();
  }

  @Test(timeout = 4000)
  public void test78()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.insertMarkerToFormattingElements();
      htmlTreeBuilder0.clearFormattingElementsToLastMarker();
  }

  @Test(timeout = 4000)
  public void test79()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.removeFromActiveFormattingElements((Element) null);
  }

  @Test(timeout = 4000)
  public void test80()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("td", "td");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.removeFromActiveFormattingElements(document0);
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test81()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("r9I[`eha$f>S_", "&c'j");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      Element element0 = htmlTreeBuilder0.getActiveFormattingElement("head");
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test82()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.insertMarkerToFormattingElements();
      Element element0 = htmlTreeBuilder0.getActiveFormattingElement("AfterBody");
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test83()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("r9I[`eha$f>S_", "&c'j");
      Element element0 = document0.head();
      htmlTreeBuilder0.pushActiveFormattingElements(element0);
      Element element1 = htmlTreeBuilder0.getActiveFormattingElement("head");
      assertEquals("&c'j", element1.baseUri());
      assertEquals(0, element1.siblingIndex());
      assertNotNull(element1);
  }

  @Test(timeout = 4000)
  public void test84()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("Rdd", "Rdd");
      Token.StartTag token_StartTag0 = new Token.StartTag("table");
      Element element0 = htmlTreeBuilder0.insert(token_StartTag0);
      element0.remove();
      htmlTreeBuilder0.insertInFosterParent(element0);
      assertEquals(1, element0.siblingIndex());
  }
}