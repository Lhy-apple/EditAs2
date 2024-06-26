/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:17:06 GMT 2023
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
import org.jsoup.nodes.CDataNode;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.FormElement;
import org.jsoup.nodes.Node;
import org.jsoup.nodes.PseudoTextElement;
import org.jsoup.parser.HtmlTreeBuilder;
import org.jsoup.parser.HtmlTreeBuilderState;
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
      Parser parser0 = new Parser(htmlTreeBuilder0);
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      ParseSettings parseSettings0 = xmlTreeBuilder0.defaultSettings();
      CDataNode cDataNode0 = new CDataNode("D$V`zQ");
      Attributes attributes0 = cDataNode0.attributes();
      Tag tag0 = Tag.valueOf("D$V`zQ", parseSettings0);
      PseudoTextElement pseudoTextElement0 = new PseudoTextElement(tag0, "D$V`zQ", attributes0);
      parser0.parseFragmentInput("k&&bQsBap@RZx", pseudoTextElement0, "H|}'PK8vJM&()6");
      Token.StartTag token_StartTag0 = new Token.StartTag();
      Token.StartTag token_StartTag1 = token_StartTag0.nameAttr("frameset", attributes0);
      htmlTreeBuilder0.insertEmpty(token_StartTag1);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      ParseSettings parseSettings0 = htmlTreeBuilder0.defaultSettings();
      Tag tag0 = Tag.valueOf("html", parseSettings0);
      CDataNode cDataNode0 = new CDataNode("button");
      Attributes attributes0 = cDataNode0.attributes();
      PseudoTextElement pseudoTextElement0 = new PseudoTextElement(tag0, "select", attributes0);
      List<Node> list0 = parser0.parseFragmentInput("\"GUKKod", pseudoTextElement0, "\"GUKKod");
      assertEquals(2, list0.size());
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
      Document document0 = Parser.parseBodyFragment("osM)Ww", "osM)Ww");
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseFragmentInput("Tag cannot be self closing; not a void tag", document0, "Tag cannot be self closing; not a void tag");
      CDataNode cDataNode0 = new CDataNode("Tag cannot be self closing; not a void tag");
      Attributes attributes0 = cDataNode0.attributes();
      Tokeniser tokeniser0 = htmlTreeBuilder0.tokeniser;
      Token.StartTag token_StartTag0 = tokeniser0.startPending;
      token_StartTag0.nameAttr("osM)Ww", attributes0);
      FormElement formElement0 = htmlTreeBuilder0.insertForm(token_StartTag0, false);
      assertEquals("osm)ww", formElement0.tagName());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      ArrayList<Element> arrayList0 = htmlTreeBuilder0.getStack();
      assertNull(arrayList0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
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
      Parser parser0 = Parser.htmlParser();
      ParseSettings parseSettings0 = htmlTreeBuilder0.defaultSettings();
      Tag tag0 = Tag.valueOf("tfoot", parseSettings0);
      PseudoTextElement pseudoTextElement0 = new PseudoTextElement(tag0, "tfoot", (Attributes) null);
      List<Node> list0 = parser0.parseFragmentInput("tfoot", pseudoTextElement0, "tfoot");
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.generateImpliedEndTags();
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.insertMarkerToFormattingElements();
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
      FormElement formElement0 = htmlTreeBuilder0.getFormElement();
      assertNull(formElement0);
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.state();
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
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
      Document document0 = Document.createShell("<%f&#/E!YN%R;");
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.isInActiveFormattingElements(document0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.HtmlTreeBuilder", e);
      }
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
  public void test21()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      List<Node> list0 = parser0.parseFragmentInput("gN<!", (Element) null, "tr");
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.inTableScope((String) null);
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
  public void test24()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      Tag tag0 = Tag.valueOf("style");
      Attributes attributes0 = new Attributes();
      PseudoTextElement pseudoTextElement0 = new PseudoTextElement(tag0, "EOF", attributes0);
      parser0.parseFragmentInput("EOF", pseudoTextElement0, "style");
      htmlTreeBuilder0.clearStackToTableRowContext();
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      Tag tag0 = Tag.valueOf("Ea:]<ma\u0003?.@A}^U");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "Ea:]<ma\u0003?.@A}^U", attributes0);
      List<Node> list0 = parser0.parseFragmentInput("Ea:]<ma\u0003?.@A}^U", formElement0, "J*1imcy'q $m&>n-");
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Parser.parseBodyFragment("asc8ii", "asc8ii");
      htmlTreeBuilder0.maybeSetBaseUri(document0);
      assertEquals("asc8ii", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.xmlParser();
      parser0.setTrackErrors(576);
      parser0.setTreeBuilder(htmlTreeBuilder0);
      HtmlTreeBuilderState htmlTreeBuilderState0 = HtmlTreeBuilderState.AfterBody;
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.error(htmlTreeBuilderState0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.HtmlTreeBuilder", e);
      }
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
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseFragmentInput("osM)Ww", (Element) null, "Tag cannot be self closing; not a void tag");
      Token.StartTag token_StartTag0 = new Token.StartTag();
      CDataNode cDataNode0 = new CDataNode("Tvj");
      Attributes attributes0 = cDataNode0.attributes();
      token_StartTag0.nameAttr("Tvj", attributes0);
      FormElement formElement0 = htmlTreeBuilder0.insertForm(token_StartTag0, true);
      assertEquals(1, formElement0.siblingIndex());
      assertEquals("Tag cannot be self closing; not a void tag", formElement0.baseUri());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Parser.parseBodyFragment("osM)Ww", "osM)Ww");
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseFragmentInput("osM)Ww", document0, "osM)Ww");
      Token.CData token_CData0 = new Token.CData("osM)Ww");
      htmlTreeBuilder0.insert(token_CData0);
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      Document document0 = Document.createShell("J*1imcy'q $m&>n-");
      parser0.parseFragmentInput("J*1imcy'q $m&>n-", document0, "J*1imcy'q $m&>n-");
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
  public void test32()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Parser.parseBodyFragment("osM)Ww", "osM)Ww");
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseFragmentInput("osM)Ww", document0, "osM)Ww");
      Element element0 = htmlTreeBuilder0.currentElement();
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.aboveOnStack(element0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseFragmentInput("thead", (Element) null, "[kIuWkmo-8ak");
      Element element0 = htmlTreeBuilder0.getFromStack("body");
      assertEquals("[kIuWkmo-8ak", element0.baseUri());
      assertNotNull(element0);
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      Parser parser1 = parser0.setTreeBuilder(htmlTreeBuilder0);
      ParseSettings parseSettings0 = htmlTreeBuilder0.defaultSettings();
      Tag tag0 = Tag.valueOf("select", parseSettings0);
      CDataNode cDataNode0 = new CDataNode("button");
      Attributes attributes0 = cDataNode0.attributes();
      PseudoTextElement pseudoTextElement0 = new PseudoTextElement(tag0, "5#G&,%gYMG7<", attributes0);
      List<Node> list0 = parser1.parseFragmentInput("4;0'-t-|Y9-#3", pseudoTextElement0, "0~/d}U/OX^p");
      assertEquals(1, list0.size());
      
      Document document0 = Parser.parseBodyFragmentRelaxed("0~/d}U/OX^p", "5#G&,%gYMG7<");
      boolean boolean0 = htmlTreeBuilder0.removeFromStack(document0);
      assertEquals(1, document0.childNodeSize());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseFragmentInput("tr", (Element) null, "tr");
      htmlTreeBuilder0.popStackToClose("np{Y");
      boolean boolean0 = htmlTreeBuilder0.inButtonScope("tr");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseFragmentInput("?rg.jsotp.select.Evaluato9$AttributeWithVal|eStart|ng", (Element) null, "body");
      htmlTreeBuilder0.popStackToClose("body");
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseFragmentInput("J*1imcy'q $m&>n-", (Element) null, "J*1imcy'q $m&>n-");
      String[] stringArray0 = new String[2];
      stringArray0[0] = "!b";
      stringArray0[1] = "!b";
      htmlTreeBuilder0.popStackToClose(stringArray0);
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Parser.parseBodyFragment("M/e:.+KL", "html");
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseFragmentInput(" $ j ", document0, "M/e:.+KL");
      String[] stringArray0 = new String[3];
      stringArray0[1] = "html";
      htmlTreeBuilder0.popStackToClose(stringArray0);
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseFragmentInput("tr", (Element) null, "tr");
      htmlTreeBuilder0.popStackToBefore("cY#X&hQdt");
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Parser.parseBodyFragment("command", "command");
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseFragmentInput("command", document0, "command");
      htmlTreeBuilder0.popStackToBefore("html");
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseFragmentInput("J*1imcy'q $m&>n-", (Element) null, "J*1imcy'q $m&>n-");
      htmlTreeBuilder0.clearStackToTableRowContext();
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Parser.parseBodyFragment("command", "command");
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseFragmentInput("command", document0, "command");
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
  public void test43()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseFragmentInput("J*1imcy'q $m&>n-", (Element) null, "J*1imcy'q $m&>n-");
      Element element0 = htmlTreeBuilder0.insertStartTag("J*1imcy'q $m&>n-");
      htmlTreeBuilder0.insertOnStackAfter(element0, (Element) null);
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseFragmentInput("org.jsoup.select.Evaluator$AttributeWithValueStarting", (Element) null, "org.jsoup.select.Evaluator$AttributeWithValueStarting");
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
  public void test45()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseFragmentInput("_D6(OaKIg88wSu)}D%S", (Element) null, "_D6(OaKIg88wSu)}D%S");
      htmlTreeBuilder0.resetInsertionMode();
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      Element element0 = new Element("th");
      List<Node> list0 = parser0.parseFragmentInput("d$v`zq", element0, "^+");
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      Document document0 = Parser.parse("tr", "tr");
      Element element0 = document0.createElement("tr");
      List<Node> list0 = parser0.parseFragmentInput("HY", element0, "tV");
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Parser.parseBodyFragment("osM)Ww", "0d");
      Parser parser0 = new Parser(htmlTreeBuilder0);
      Element element0 = document0.createElement("thead");
      List<Node> list0 = parser0.parseFragmentInput("0d", element0, "osM)Ww");
      assertFalse(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      Element element0 = new Element("table");
      List<Node> list0 = parser0.parseFragmentInput("fmmrzv3b3}\u0001ma>mr[t", element0, "table");
      assertFalse(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      Document document0 = Document.createShell("J*1imcy'q $m&>n-");
      Element element0 = document0.head();
      List<Node> list0 = parser0.parseFragmentInput("J*1imcy'q $m&>n-", element0, "J*1imcy'q $m&>n-");
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Parser.parseBodyFragment("M/e:.+KL", "html");
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseFragmentInput("    ", document0, "M/e:.+KL");
      boolean boolean0 = htmlTreeBuilder0.inButtonScope("html");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseFragmentInput("org.jsoup.select.Evaluator$AttributeWithValueStarting", (Element) null, "org.jsoup.select.Evaluator$AttributeWithValueStarting");
      htmlTreeBuilder0.inListItemScope("org.jsoup.select.Evaluator$AttributeWithValueStarting");
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      List<Node> list0 = parser0.parseFragmentInput("org.jsoup.select.Evaluator$AttributeWithValueStarting", (Element) null, "org.jsoup.select.Evaluator$AttributeWithValueStarting");
      assertEquals(1, list0.size());
      
      boolean boolean0 = htmlTreeBuilder0.inScope("org.jsoup.select.Evaluator$AttributeWithValueStarting");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseFragmentInput("tr", (Element) null, "tr");
      htmlTreeBuilder0.popStackToClose("np{Y");
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.inSelectScope("np{Y");
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
      Parser parser0 = new Parser(htmlTreeBuilder0);
      List<Node> list0 = parser0.parseFragmentInput("org.jsoup.select.Evaluator$AttributeWithValueStarti~g", (Element) null, "org.jsoup.select.Evaluator$AttributeWithValueStarti~g");
      assertEquals(1, list0.size());
      
      boolean boolean0 = htmlTreeBuilder0.inSelectScope("org.jsoup.select.Evaluator$AttributeWithValueStarti~g");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Parser.parseBodyFragment("M/e:.+KL", "html");
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseFragmentInput("M/e:.+KL", document0, "M/e:.+KL");
      htmlTreeBuilder0.generateImpliedEndTags("html");
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Parser.parseBodyFragment("M/e:.+KL", "html");
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseFragmentInput("    ", document0, "M/e:.+KL");
      htmlTreeBuilder0.generateImpliedEndTags("    ");
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseFragmentInput("_D6(OaKIg88wSu)}D%S", (Element) null, "_D6(OaKIg88wSu)}D%S");
      htmlTreeBuilder0.pushActiveFormattingElements((Element) null);
      Element element0 = htmlTreeBuilder0.lastFormattingElement();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Parser.parseBodyFragment("osM)Ww", "osM)Ww");
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseFragmentInput("osM)Ww", document0, "osM)Ww");
      Element element0 = htmlTreeBuilder0.removeLastFormattingElement();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      ParseSettings parseSettings0 = xmlTreeBuilder0.defaultSettings();
      CDataNode cDataNode0 = new CDataNode("D$V`zQ");
      Attributes attributes0 = cDataNode0.attributes();
      Tag tag0 = Tag.valueOf("D$V`zQ", parseSettings0);
      PseudoTextElement pseudoTextElement0 = new PseudoTextElement(tag0, "D$V`zQ", attributes0);
      parser0.parseFragmentInput("k&&bQsBap@RZx", pseudoTextElement0, "H|}'PK8vJM&()6");
      htmlTreeBuilder0.pushActiveFormattingElements(pseudoTextElement0);
      htmlTreeBuilder0.pushActiveFormattingElements(pseudoTextElement0);
      htmlTreeBuilder0.pushActiveFormattingElements(pseudoTextElement0);
      htmlTreeBuilder0.pushActiveFormattingElements(pseudoTextElement0);
      assertFalse(pseudoTextElement0.isBlock());
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseFragmentInput("_D6(OaKIg88wSu)}D%S", (Element) null, "_D6(OaKIg88wSu)}D%S");
      htmlTreeBuilder0.pushActiveFormattingElements((Element) null);
      htmlTreeBuilder0.pushActiveFormattingElements((Element) null);
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Parser.parseBodyFragment("osM)Ww", "osM)Ww");
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseFragmentInput("osM)Ww", document0, "osM)Ww");
      Attributes attributes0 = new Attributes();
      ParseSettings parseSettings0 = ParseSettings.htmlDefault;
      Tag tag0 = Tag.valueOf("xhin^{B>\"e", parseSettings0);
      PseudoTextElement pseudoTextElement0 = new PseudoTextElement(tag0, "xhin^{B>\"e", attributes0);
      htmlTreeBuilder0.pushActiveFormattingElements(pseudoTextElement0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseFragmentInput("org.jsoup.select.Evaluator$AttributeWithValueStarti~g", (Element) null, "org.jsoup.select.Evaluator$AttributeWithValueStarti~g");
      htmlTreeBuilder0.pushActiveFormattingElements((Element) null);
      htmlTreeBuilder0.clearFormattingElementsToLastMarker();
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      ParseSettings parseSettings0 = ParseSettings.htmlDefault;
      CDataNode cDataNode0 = new CDataNode("org.jsoup.select.Evaluator$AttributeWithValueStarti~g");
      Attributes attributes0 = cDataNode0.attributes();
      Tag tag0 = Tag.valueOf("k&&bQsBap@RZx", parseSettings0);
      PseudoTextElement pseudoTextElement0 = new PseudoTextElement(tag0, "h6", attributes0);
      parser0.parseFragmentInput("org.jsoup.select.Evaluator$AttributeWithValueStarti~g", (Element) null, "org.jsoup.select.Evaluator$AttributeWithValueStarti~g");
      htmlTreeBuilder0.pushActiveFormattingElements(pseudoTextElement0);
      htmlTreeBuilder0.clearFormattingElementsToLastMarker();
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Parser.parseBodyFragment("osM)Ww", "0d");
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseFragmentInput("0d", document0, "osM)Ww");
      htmlTreeBuilder0.removeFromActiveFormattingElements(document0);
      assertEquals("0d", document0.location());
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Parser.parseBodyFragment("osM)Ww", "0d");
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseFragmentInput("0d", document0, "osM)Ww");
      Element element0 = htmlTreeBuilder0.getActiveFormattingElement("Y.`4O^RP*Pu:lM>");
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Parser.parseBodyFragment("osM)Ww", "0d");
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseFragmentInput("0d", document0, "osM)Ww");
      htmlTreeBuilder0.insertInFosterParent(document0);
      assertEquals("#root", document0.tagName());
  }
}
