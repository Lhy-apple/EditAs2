/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:15:37 GMT 2023
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
import org.jsoup.nodes.FormElement;
import org.jsoup.nodes.Node;
import org.jsoup.nodes.PseudoTextElement;
import org.jsoup.parser.CharacterReader;
import org.jsoup.parser.HtmlTreeBuilder;
import org.jsoup.parser.ParseErrorList;
import org.jsoup.parser.ParseSettings;
import org.jsoup.parser.Parser;
import org.jsoup.parser.Tag;
import org.jsoup.parser.Token;
import org.jsoup.parser.Tokeniser;
import org.jsoup.parser.XmlTreeBuilder;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class HtmlTreeBuilder_ESTest extends HtmlTreeBuilder_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      StringReader stringReader0 = new StringReader("org.jsoup.select.Evaluator$Attribute");
      htmlTreeBuilder0.parse(stringReader0, "org.jsoup.select.Evaluator$Attribute", parser0);
      Tokeniser tokeniser0 = htmlTreeBuilder0.tokeniser;
      Token.StartTag token_StartTag0 = tokeniser0.startPending;
      Attributes attributes0 = new Attributes();
      Token.StartTag token_StartTag1 = token_StartTag0.nameAttr("j`S@RX", attributes0);
      htmlTreeBuilder0.insertForm(token_StartTag1, false);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Element element0 = htmlTreeBuilder0.getHeadElement();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.xmlParser();
      StringReader stringReader0 = new StringReader("org.jsoup.select.Evluator$Attrib&te");
      Document document0 = htmlTreeBuilder0.parse(stringReader0, "org.jsoup.select.Evluator$Attrib&te", parser0);
      assertEquals(1, document0.childNodeSize());
      
      boolean boolean0 = htmlTreeBuilder0.processEndTag("org.jsoup.select.Evluator$Attrib&te");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
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
  public void test04()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.inListItemScope(",sL1X/9:G");
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
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      StringReader stringReader0 = new StringReader("div");
      htmlTreeBuilder0.parse(stringReader0, "div", parser0);
      htmlTreeBuilder0.processStartTag("marquee");
      htmlTreeBuilder0.resetInsertionMode();
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
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
      htmlTreeBuilder0.state();
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.getDocument();
      assertNull(document0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.inScope("v}W,_Z|e");
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
      String string0 = htmlTreeBuilder0.getBaseUri();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      StringReader stringReader0 = new StringReader("org.jsoup.select.Evaluator$Attribute");
      Document document0 = htmlTreeBuilder0.parse(stringReader0, "org.jsoup.select.Evaluator$Attribute", parser0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.processStartTag("org.jsoup.select.Evaluator$Attribute");
      htmlTreeBuilder0.reconstructFormattingElements();
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      StringReader stringReader0 = new StringReader("org.jsoup.select.Evaluator$Attribute");
      Document document0 = htmlTreeBuilder0.parse(stringReader0, "org.jsoup.select.Evaluator$Attribute", parser0);
      boolean boolean0 = htmlTreeBuilder0.isInActiveFormattingElements(document0);
      assertFalse(boolean0);
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      boolean boolean0 = htmlTreeBuilder0.isFragmentParsing();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Parser.parseBodyFragment("!=BhtDp$9dywOkO", "!=BhtDp$9dywOkO");
      ParseSettings parseSettings0 = new ParseSettings(false, false);
      Tag tag0 = Tag.valueOf("!=BhtDp$9dywOkO", parseSettings0);
      Attributes attributes0 = new Attributes();
      PseudoTextElement pseudoTextElement0 = new PseudoTextElement(tag0, "!=BhtDp$9dywOkO", attributes0);
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.replaceActiveFormattingElement(document0, pseudoTextElement0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.HtmlTreeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.setFosterInserts(false);
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
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
  public void test20()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.xmlParser();
      Document document0 = Parser.parseBodyFragment("tr", "ul");
      List<Node> list0 = htmlTreeBuilder0.parseFragment("rIi+9ydPp&LL<!2|Ot", document0, "Aue~o8J,DqZ%j?c", parser0);
      assertEquals(2, list0.size());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.inTableScope("tfoot");
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
      Parser parser0 = Parser.htmlParser();
      List<Node> list0 = parser0.parseFragmentInput("", (Element) null, "org.jsoup.select.Evaluator$Attribute");
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Tag tag0 = Tag.valueOf("style", (ParseSettings) null);
      Attributes attributes0 = new Attributes();
      Element element0 = new Element(tag0, "style", attributes0);
      Parser parser0 = new Parser(htmlTreeBuilder0);
      List<Node> list0 = parser0.parseFragmentInput("style", element0, "style");
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      ParseSettings parseSettings0 = xmlTreeBuilder0.defaultSettings();
      Tag tag0 = Tag.valueOf("9QM,*f]", parseSettings0);
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "table", attributes0);
      htmlTreeBuilder0.maybeSetBaseUri(formElement0);
      assertEquals(0, formElement0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Parser parser0 = Parser.htmlParser();
      Attributes attributes0 = new Attributes();
      ParseSettings parseSettings0 = ParseSettings.preserveCase;
      Tag tag0 = Tag.valueOf("thead", parseSettings0);
      Element element0 = new Element(tag0, "thead", attributes0);
      parser0.setTrackErrors(100);
      List<Node> list0 = parser0.parseFragmentInput("s1x2Fq~qvNbuMv", element0, "s1x2Fq~qvNbuMv");
      assertFalse(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      StringReader stringReader0 = new StringReader("org.jsoup.select.Evaluator$Attribute");
      htmlTreeBuilder0.parse(stringReader0, "org.jsoup.select.Evaluator$Attribute", parser0);
      Attributes attributes0 = new Attributes();
      Token.StartTag token_StartTag0 = new Token.StartTag();
      Token.StartTag token_StartTag1 = token_StartTag0.nameAttr("input", attributes0);
      Element element0 = htmlTreeBuilder0.insertEmpty(token_StartTag1);
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      StringReader stringReader0 = new StringReader("org.jsoup.select.Evaluator$Attribute");
      htmlTreeBuilder0.parse(stringReader0, "org.jsoup.select.Evaluator$Attribute", parser0);
      CharacterReader characterReader0 = new CharacterReader(stringReader0, 100);
      ParseErrorList parseErrorList0 = parser0.getErrors();
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      Token.StartTag token_StartTag0 = tokeniser0.startPending;
      Attributes attributes0 = new Attributes();
      token_StartTag0.nameAttr("body", attributes0);
      FormElement formElement0 = htmlTreeBuilder0.insertForm(token_StartTag0, true);
      assertEquals(1, formElement0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      StringReader stringReader0 = new StringReader("org.jsoup.select.Evaluator$Attribute");
      htmlTreeBuilder0.parse(stringReader0, "org.jsoup.select.Evaluator$Attribute", parser0);
      Token.CData token_CData0 = new Token.CData("1Azp-UkMHR0pyh@/");
      htmlTreeBuilder0.insert(token_CData0);
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Parser.parseBodyFragment("!=BhtDp$9dywOkO", "!=BhtDp$9dywOkO");
      Parser parser0 = Parser.htmlParser();
      htmlTreeBuilder0.parseFragment("!=BhtDp$9dywOkO", document0, "!=BhtDp$9dywOkO", parser0);
      htmlTreeBuilder0.insertInFosterParent(document0);
      assertEquals(1, document0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      Tag tag0 = Tag.valueOf("C]}|H^h=cG");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "<", attributes0);
      List<Node> list0 = htmlTreeBuilder0.parseFragment("table", formElement0, "@I@{$E0R#[=NF,", parser0);
      assertEquals(1, list0.size());
      
      boolean boolean0 = htmlTreeBuilder0.removeFromStack(formElement0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Parser.parseBodyFragment("!=BhtDp$9dcywOkO", "!=BhtDp$9dcywOkO");
      Parser parser0 = Parser.htmlParser();
      htmlTreeBuilder0.parseFragment("wTc.sz0fC", document0, "!=BhtDp$9dcywOkO", parser0);
      htmlTreeBuilder0.popStackToClose("wTc.sz0fC");
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      Document document0 = Parser.parseBodyFragment("InRow", "e,[Y9gX%");
      htmlTreeBuilder0.parseFragment("org.jsoup.select.evaluator$attribute", document0, "e,[Y9gX%", parser0);
      htmlTreeBuilder0.popStackToClose("html");
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Tag tag0 = Tag.valueOf("html");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "table", attributes0);
      Parser parser0 = Parser.xmlParser();
      htmlTreeBuilder0.parseFragment("CF6{mLUxE:", formElement0, "html", parser0);
      String[] stringArray0 = new String[1];
      stringArray0[0] = "html";
      htmlTreeBuilder0.popStackToClose(stringArray0);
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Parser.parseBodyFragment("!=BhtDp$9dywOkO", "!=BhtDp$9dywOkO");
      Parser parser0 = Parser.htmlParser();
      htmlTreeBuilder0.parseFragment("!=BhtDp$9dywOkO", document0, "!=BhtDp$9dywOkO", parser0);
      htmlTreeBuilder0.popStackToBefore("'lWg");
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      StringReader stringReader0 = new StringReader("org.jsoup.select.Evaluator$Attribute");
      htmlTreeBuilder0.parse(stringReader0, "org.jsoup.select.Evaluator$Attribute", parser0);
      htmlTreeBuilder0.popStackToBefore("body");
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      StringReader stringReader0 = new StringReader("org.jsoup.select.Evaluator$Attribute");
      htmlTreeBuilder0.parse(stringReader0, "org.jsoup.select.Evaluator$Attribute", parser0);
      String[] stringArray0 = new String[3];
      stringArray0[0] = "org.jsoup.select.Evaluator$Attribute";
      stringArray0[1] = "org.jsoup.select.Evaluator$Attribute";
      htmlTreeBuilder0.popStackToClose(stringArray0);
      htmlTreeBuilder0.clearStackToTableContext();
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      StringReader stringReader0 = new StringReader("org.jsoup.select.Evaluator$Attribute");
      htmlTreeBuilder0.parse(stringReader0, "org.jsoup.select.Evaluator$Attribute", parser0);
      htmlTreeBuilder0.clearStackToTableContext();
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      StringReader stringReader0 = new StringReader("9QM,*f]");
      htmlTreeBuilder0.parse(stringReader0, "9QM,*f]", parser0);
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      ParseSettings parseSettings0 = xmlTreeBuilder0.defaultSettings();
      Tag tag0 = Tag.valueOf("9QM,*f]", parseSettings0);
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "table", attributes0);
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.aboveOnStack(formElement0);
        fail("Expecting exception: AssertionError");
      
      } catch(AssertionError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Document.createShell(": ");
      Parser parser0 = new Parser(htmlTreeBuilder0);
      htmlTreeBuilder0.parseFragment("^gk[2", document0, "^gk[2", parser0);
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
  public void test40()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Parser.parseBodyFragment("!=BhtDp$9dywOkO", "!=BhtDp$9dywOkO");
      Parser parser0 = Parser.htmlParser();
      htmlTreeBuilder0.parseFragment("!=BhtDp$9dywOkO", document0, "!=BhtDp$9dywOkO", parser0);
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.replaceOnStack(document0, document0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Must be true
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      StringReader stringReader0 = new StringReader("org.jsoup.select.Evaluator$Attribute");
      Document document0 = htmlTreeBuilder0.parse(stringReader0, "org.jsoup.select.Evaluator$Attribute", parser0);
      Element element0 = htmlTreeBuilder0.insertStartTag("org.jsoup.select.Evaluator$Attribute");
      htmlTreeBuilder0.replaceOnStack(element0, document0);
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      StringReader stringReader0 = new StringReader("org.jsoup.select.Evaluator$Attribute");
      htmlTreeBuilder0.parse(stringReader0, "org.jsoup.select.Evaluator$Attribute", parser0);
      String[] stringArray0 = new String[3];
      stringArray0[0] = "org.jsoup.select.Evaluator$Attribute";
      stringArray0[1] = "org.jsoup.select.Evaluator$Attribute";
      htmlTreeBuilder0.popStackToClose(stringArray0);
      htmlTreeBuilder0.resetInsertionMode();
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Tag tag0 = Tag.valueOf("html");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "table", attributes0);
      Parser parser0 = Parser.xmlParser();
      htmlTreeBuilder0.parseFragment("CF6{mLUxE:", formElement0, "html", parser0);
      htmlTreeBuilder0.resetInsertionMode();
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Tag tag0 = Tag.valueOf("select", (ParseSettings) null);
      Attributes attributes0 = new Attributes();
      Element element0 = new Element(tag0, "select", attributes0);
      Parser parser0 = new Parser(htmlTreeBuilder0);
      List<Node> list0 = parser0.parseFragmentInput("org.jsoup.select.Evaluator$Attribute", element0, "org.jsoup.select.Evaluator$Attribute");
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Tag tag0 = Tag.valueOf("th", (ParseSettings) null);
      Attributes attributes0 = new Attributes();
      Element element0 = new Element(tag0, "th", attributes0);
      Parser parser0 = new Parser(htmlTreeBuilder0);
      List<Node> list0 = parser0.parseFragmentInput("th", element0, "th");
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Tag tag0 = Tag.valueOf("tr", (ParseSettings) null);
      FormElement formElement0 = new FormElement(tag0, "wkLA{qD7H", attributes0);
      Parser parser0 = Parser.htmlParser();
      List<Node> list0 = parser0.parseFragmentInput("I*2Ls&!kR3_h^zo.GyG", formElement0, "tr");
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Parser parser0 = Parser.htmlParser();
      Tag tag0 = Tag.valueOf("caption");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "org.jsoup.select.Evaluator$Attribute", attributes0);
      List<Node> list0 = parser0.parseFragmentInput("details", formElement0, "org.jsoup.select.Evaluator$Attribute");
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      StringReader stringReader0 = new StringReader("org.jsoup.select.Evaluator$Attribute");
      Document document0 = htmlTreeBuilder0.parse(stringReader0, "org.jsoup.select.Evaluator$Attribute", parser0);
      Element element0 = document0.head();
      List<Node> list0 = htmlTreeBuilder0.parseFragment("org.jsoup.select.Evaluator$Attribute", element0, "MX6Cgf=Np?wYx~Sx", parser0);
      assertEquals(0, element0.childNodeSize());
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Parser parser0 = Parser.htmlParser();
      Tag tag0 = Tag.valueOf("frameset");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "frameset", attributes0);
      List<Node> list0 = parser0.parseFragmentInput("frameset", formElement0, "org.jsoup.select.Evaluator$Attribute");
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      StringReader stringReader0 = new StringReader("org.jsoup.select.Evaluator$Attribute");
      Document document0 = htmlTreeBuilder0.parse(stringReader0, "org.jsoup.select.Evaluator$Attribute", parser0);
      assertEquals(1, document0.childNodeSize());
      
      String[] stringArray0 = new String[3];
      stringArray0[0] = "org.jsoup.select.Evaluator$Attribute";
      stringArray0[1] = "org.jsoup.select.Evaluator$Attribute";
      htmlTreeBuilder0.popStackToClose(stringArray0);
      boolean boolean0 = htmlTreeBuilder0.inButtonScope("mZ/MUP(d}fOB");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      StringReader stringReader0 = new StringReader("org.jsoup.select.Evaluator$Attribute");
      Document document0 = htmlTreeBuilder0.parse(stringReader0, "org.jsoup.select.Evaluator$Attribute", parser0);
      assertEquals(1, document0.childNodeSize());
      
      boolean boolean0 = htmlTreeBuilder0.inButtonScope("body");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      StringReader stringReader0 = new StringReader("org.jsoup.select.Evaluator$Attribute");
      Document document0 = htmlTreeBuilder0.parse(stringReader0, "org.jsoup.select.Evaluator$Attribute", parser0);
      assertEquals(1, document0.childNodeSize());
      
      String[] stringArray0 = new String[6];
      stringArray0[0] = "org.jsoup.select.Evaluator$Attribute";
      stringArray0[2] = "org.jsoup.select.Evaluator$Attribute";
      boolean boolean0 = htmlTreeBuilder0.inScope(stringArray0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      StringReader stringReader0 = new StringReader("org.jsoup.select.Evaluator$Attribute");
      Document document0 = htmlTreeBuilder0.parse(stringReader0, "org.jsoup.select.Evaluator$Attribute", parser0);
      assertEquals(1, document0.childNodeSize());
      
      boolean boolean0 = htmlTreeBuilder0.inButtonScope("tfoot");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      StringReader stringReader0 = new StringReader("org.jsoup.select.Evaluator$Attribute");
      htmlTreeBuilder0.parse(stringReader0, "org.jsoup.select.Evaluator$Attribute", parser0);
      String[] stringArray0 = new String[3];
      stringArray0[0] = "org.jsoup.select.Evaluator$Attribute";
      stringArray0[1] = "org.jsoup.select.Evaluator$Attribute";
      htmlTreeBuilder0.popStackToClose(stringArray0);
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.inSelectScope("|EHX':$<k=k:S3k");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Should not be reachable
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Parser.parseBodyFragment("!=BhtDp$9dcywOkO", "!=BhtDp$9dcywOkO");
      Parser parser0 = Parser.htmlParser();
      List<Node> list0 = htmlTreeBuilder0.parseFragment("wTc.sz0fC", document0, "!=BhtDp$9dcywOkO", parser0);
      assertEquals(1, list0.size());
      
      boolean boolean0 = htmlTreeBuilder0.inSelectScope("&gt;");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      Tag tag0 = Tag.valueOf("C]}|H^h=cG");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "<", attributes0);
      List<Node> list0 = htmlTreeBuilder0.parseFragment("table", formElement0, "@I@{$E0R#[=NF,", parser0);
      assertEquals(1, list0.size());
      
      boolean boolean0 = htmlTreeBuilder0.inSelectScope("html");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Parser.parseBodyFragment("!=BhtDp$9dywOkO", "!=BhtDp$9dywOkO");
      Parser parser0 = Parser.htmlParser();
      htmlTreeBuilder0.parseFragment("!=BhtDp$9dywOkO", document0, "!=BhtDp$9dywOkO", parser0);
      htmlTreeBuilder0.generateImpliedEndTags("html");
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      StringReader stringReader0 = new StringReader("org.jsoup.select.Evaluator$Attribute");
      htmlTreeBuilder0.parse(stringReader0, "org.jsoup.select.Evaluator$Attribute", parser0);
      htmlTreeBuilder0.generateImpliedEndTags("org.jsoup.select.Evaluator$Attribute");
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      Document document0 = Parser.parseBodyFragment("InRow", "e,[Y9gX%");
      htmlTreeBuilder0.parseFragment("org.jsoup.select.evaluator$attribute", document0, "e,[Y9gX%", parser0);
      Element element0 = htmlTreeBuilder0.removeLastFormattingElement();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      StringReader stringReader0 = new StringReader("org.jsoup.select.Evaluator$Attribute");
      Document document0 = htmlTreeBuilder0.parse(stringReader0, "org.jsoup.select.Evaluator$Attribute", parser0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.clearFormattingElementsToLastMarker();
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      StringReader stringReader0 = new StringReader("9QM,*f]");
      Document document0 = htmlTreeBuilder0.parse(stringReader0, "9QM,*f]", parser0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      StringReader stringReader0 = new StringReader("div");
      Document document0 = htmlTreeBuilder0.parse(stringReader0, "div", parser0);
      htmlTreeBuilder0.processStartTag("marquee");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.reconstructFormattingElements();
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      StringReader stringReader0 = new StringReader("9QM,*f]");
      Document document0 = htmlTreeBuilder0.parse(stringReader0, "9QM,*f]", parser0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      Attributes attributes0 = new Attributes();
      Tokeniser tokeniser0 = htmlTreeBuilder0.tokeniser;
      Token.StartTag token_StartTag0 = tokeniser0.startPending;
      Token.StartTag token_StartTag1 = token_StartTag0.nameAttr(".tCZZ&t,^Z {JDAW[Z*", attributes0);
      Element element0 = htmlTreeBuilder0.insertEmpty(token_StartTag1);
      htmlTreeBuilder0.pushActiveFormattingElements(element0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      StringReader stringReader0 = new StringReader("9QM,*f]");
      Document document0 = htmlTreeBuilder0.parse(stringReader0, "9QM,*f]", parser0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.processStartTag("9QM,*f]");
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      StringReader stringReader0 = new StringReader("org.jsoup.select.Evaluator$Attribute");
      Document document0 = htmlTreeBuilder0.parse(stringReader0, "org.jsoup.select.Evaluator$Attribute", parser0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.processStartTag("org.jsoup.select.Evaluator$Attribute");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.reconstructFormattingElements();
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      StringReader stringReader0 = new StringReader("thead");
      htmlTreeBuilder0.parse(stringReader0, "org.jsoup.select.Evaluator$Attribute", parser0);
      htmlTreeBuilder0.pushActiveFormattingElements((Element) null);
      htmlTreeBuilder0.clearFormattingElementsToLastMarker();
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Parser.parseBodyFragment("!=BhtDp$9dcywOkO", "e,[Y9gX%");
      Parser parser0 = Parser.htmlParser();
      htmlTreeBuilder0.parseFragment("wTc.sz0fC", document0, "!=BhtDp$9dcywOkO", parser0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.removeFromActiveFormattingElements(document0);
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Parser.parseBodyFragment("!=BhtDp$9dcywOkO", "e[Y9gX%");
      Parser parser0 = Parser.htmlParser();
      htmlTreeBuilder0.parseFragment("wTc.sz0fC", document0, "!=BhtDp$9dcywOkO", parser0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.removeFromActiveFormattingElements((Element) null);
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Parser.parseBodyFragment("!=BhtDp$9dywOkO", "!=BhtDp$9dywOkO");
      Parser parser0 = Parser.htmlParser();
      htmlTreeBuilder0.parseFragment("!=BhtDp$9dywOkO", document0, "!=BhtDp$9dywOkO", parser0);
      htmlTreeBuilder0.getActiveFormattingElement("_d|9R1Z`)`^4W'`kj:");
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      StringReader stringReader0 = new StringReader("caption");
      Document document0 = htmlTreeBuilder0.parse(stringReader0, "caption", parser0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.getActiveFormattingElement("v}W,_Z|e");
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      StringReader stringReader0 = new StringReader("thead");
      htmlTreeBuilder0.parse(stringReader0, "org.jsoup.select.Evaluator$Attribute", parser0);
      htmlTreeBuilder0.pushActiveFormattingElements((Element) null);
      htmlTreeBuilder0.getActiveFormattingElement("thead");
  }
}