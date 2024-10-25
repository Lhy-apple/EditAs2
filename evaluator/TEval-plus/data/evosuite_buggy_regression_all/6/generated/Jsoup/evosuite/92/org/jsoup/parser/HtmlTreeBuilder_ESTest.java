/*
 * This file was automatically generated by EvoSuite
 * Wed Sep 27 00:39:30 GMT 2023
 */

package org.jsoup.parser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Attributes;
import org.jsoup.nodes.CDataNode;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.FormElement;
import org.jsoup.nodes.Node;
import org.jsoup.parser.HtmlTreeBuilder;
import org.jsoup.parser.ParseSettings;
import org.jsoup.parser.Parser;
import org.jsoup.parser.Tag;
import org.jsoup.parser.Token;
import org.jsoup.parser.XmlTreeBuilder;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class HtmlTreeBuilder_ESTest extends HtmlTreeBuilder_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
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
  public void test01()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Element element0 = htmlTreeBuilder0.getHeadElement();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Attributes attributes0 = new Attributes();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseInput("&amp;", "&amp;");
      Token.StartTag token_StartTag0 = new Token.StartTag();
      Token.StartTag token_StartTag1 = token_StartTag0.nameAttr("select", attributes0);
      htmlTreeBuilder0.insertForm(token_StartTag1, true);
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
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
  public void test04()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Attributes attributes0 = new Attributes();
      Tag tag0 = Tag.valueOf("table");
      Element element0 = new Element(tag0, "table", attributes0);
      Parser parser0 = new Parser(htmlTreeBuilder0);
      List<Node> list0 = htmlTreeBuilder0.parseFragment("7=", element0, "L0Lji/qFg<:CLq7d7YP", parser0);
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.generateImpliedEndTags();
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
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
  public void test07()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      FormElement formElement0 = htmlTreeBuilder0.getFormElement();
      assertNull(formElement0);
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
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
  public void test09()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.inButtonScope("Gb,|M(gNEp0x$&lab&");
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
      String[] stringArray0 = new String[2];
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.inScope(stringArray0);
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
      Document document0 = htmlTreeBuilder0.getDocument();
      assertNull(document0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.inScope("qOP6C{`GT1PbD)");
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
      String string0 = htmlTreeBuilder0.getBaseUri();
      assertNull(string0);
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      Document document0 = parser0.parseInput("@Ct|T", "r/\">()#$  uo|'mym");
      boolean boolean0 = htmlTreeBuilder0.isInActiveFormattingElements(document0);
      assertEquals(1, document0.childNodeSize());
      assertFalse(boolean0);
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
      Parser parser0 = new Parser(htmlTreeBuilder0);
      Document document0 = parser0.parseInput("@Ct|T", "@Ct|T");
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.replaceActiveFormattingElement(document0, (Element) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Must be true
         //
         verifyException("org.jsoup.helper.Validate", e);
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
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.insert((Token.Comment) null);
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
  public void test22()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      htmlTreeBuilder0.parseFragment("Th|R2>GJQb'", (Element) null, "n v+5_'`gpoj)p'!~p!", parser0);
      htmlTreeBuilder0.pushActiveFormattingElements((Element) null);
      htmlTreeBuilder0.reconstructFormattingElements();
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      CDataNode cDataNode0 = new CDataNode("b&?BbFp){#)]3");
      Attributes attributes0 = cDataNode0.attributes();
      Tag tag0 = Tag.valueOf("noembed");
      Element element0 = new Element(tag0, "I-x{_-|>X", attributes0);
      element0.append("THS'&$r0 qwZF");
      assertEquals(1, element0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      CDataNode cDataNode0 = new CDataNode("b&?BbFp){#)]3");
      Attributes attributes0 = cDataNode0.attributes();
      ParseSettings parseSettings0 = htmlTreeBuilder0.defaultSettings();
      Tag tag0 = Tag.valueOf("noscript", parseSettings0);
      Element element0 = new Element(tag0, "b&?bFp){#)3", attributes0);
      element0.append("b&?bFp){#)3");
      assertEquals(1, element0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      ParseSettings parseSettings0 = htmlTreeBuilder0.defaultSettings();
      Tag tag0 = Tag.valueOf("plaintext", parseSettings0);
      CDataNode cDataNode0 = new CDataNode("plaintext");
      Attributes attributes0 = cDataNode0.attributes();
      Element element0 = new Element(tag0, "plaintext", attributes0);
      element0.append("plaintext");
      assertEquals(1, element0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Tag tag0 = Tag.valueOf("rp", (ParseSettings) null);
      CDataNode cDataNode0 = new CDataNode("rp");
      Attributes attributes0 = cDataNode0.attributes();
      FormElement formElement0 = new FormElement(tag0, "rp", attributes0);
      formElement0.append("rp");
      assertEquals(1, formElement0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      Document document0 = parser0.parseInput("@Ct|T", "r/\">()#$  uo|'mym");
      htmlTreeBuilder0.maybeSetBaseUri(document0);
      assertEquals(1, document0.childNodeSize());
      assertEquals("r/\">()#$  uo|'mym", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      Parser parser1 = parser0.setTrackErrors(100);
      Document document0 = parser1.parseInput("@Ct|T", "r/\">()#$  uo|'mym");
      assertEquals(1, document0.childNodeSize());
      
      boolean boolean0 = htmlTreeBuilder0.processEndTag("r/\">()#$  uo|'mym");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Attributes attributes0 = new Attributes();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseInput("&amp;", "&amp;");
      Token.StartTag token_StartTag0 = new Token.StartTag();
      Token.StartTag token_StartTag1 = token_StartTag0.nameAttr("select", attributes0);
      Element element0 = htmlTreeBuilder0.insertEmpty(token_StartTag1);
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Attributes attributes0 = new Attributes();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseInput("numeric reference with no numerals", "numeric reference with no numerals");
      Token.StartTag token_StartTag0 = new Token.StartTag();
      Token.StartTag token_StartTag1 = token_StartTag0.nameAttr("3/?.", attributes0);
      FormElement formElement0 = htmlTreeBuilder0.insertForm(token_StartTag1, false);
      assertEquals(1, formElement0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      Document document0 = parser0.parseInput("@Ct|T", "r/\">()#$  uo|'mym");
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
      ParseSettings parseSettings0 = ParseSettings.htmlDefault;
      Tag tag0 = Tag.valueOf("address", parseSettings0);
      CDataNode cDataNode0 = new CDataNode("address");
      Attributes attributes0 = cDataNode0.attributes();
      Element element0 = new Element(tag0, "address", attributes0);
      Parser parser0 = new Parser(htmlTreeBuilder0);
      htmlTreeBuilder0.parseFragment("address", element0, "address", parser0);
      htmlTreeBuilder0.insertInFosterParent(cDataNode0);
      assertEquals(1, cDataNode0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      Document document0 = parser0.parseInput("numeric reference with no numerals", "numeric reference with no numerals");
      boolean boolean0 = htmlTreeBuilder0.removeFromStack(document0);
      assertFalse(boolean0);
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseInput("@Ct|T", "r/\">()#$  uo|'mym");
      Element element0 = htmlTreeBuilder0.insertStartTag("r/\">()#$  uo|'mym");
      boolean boolean0 = htmlTreeBuilder0.removeFromStack(element0);
      assertTrue(boolean0);
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseInput("numeric reference with no numerals", "numeric reference with no numerals");
      String[] stringArray0 = new String[6];
      stringArray0[2] = "9q$LNai/o&W''-c|~";
      stringArray0[3] = "9q$LNai/o&W''-c|~";
      stringArray0[4] = "org.jsoup.parser.HtmlTreeBuilderState";
      htmlTreeBuilder0.popStackToClose(stringArray0);
      boolean boolean0 = htmlTreeBuilder0.inListItemScope("9q$LNai/o&W''-c|~");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Document.createShell("html");
      Parser parser0 = new Parser(htmlTreeBuilder0);
      htmlTreeBuilder0.parseFragment("html", document0, "optgroup", parser0);
      String[] stringArray0 = new String[7];
      stringArray0[3] = "GT|oxLsYn";
      stringArray0[4] = "html";
      stringArray0[5] = "optgroup";
      htmlTreeBuilder0.popStackToClose(stringArray0);
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseInput("&amp;", "&amp;");
      htmlTreeBuilder0.popStackToBefore("table");
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseInput("numeric reference with no numerals", "numeric reference with no numerals");
      htmlTreeBuilder0.popStackToClose("numeric reference with no numerals");
      htmlTreeBuilder0.clearStackToTableBodyContext();
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseInput("numeric reference with no numerals", "numeric reference with no numerals");
      htmlTreeBuilder0.clearStackToTableBodyContext();
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      Document document0 = parser0.parseInput("@Ct|T", "r/\">()#$  uo|'mym");
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
  public void test41()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      Document document0 = parser0.parseInput("@Ct|T", "@Ct|T");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.replaceActiveFormattingElement(document0, (Element) null);
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseInput("numeric reference with no numerals", "numeric reference with no numerals");
      htmlTreeBuilder0.popStackToClose("numeric reference with no numerals");
      htmlTreeBuilder0.resetInsertionMode();
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      CDataNode cDataNode0 = new CDataNode("b&?BbFp){#)]3");
      Attributes attributes0 = cDataNode0.attributes();
      ParseSettings parseSettings0 = htmlTreeBuilder0.defaultSettings();
      Tag tag0 = Tag.valueOf("th", parseSettings0);
      Element element0 = new Element(tag0, "th", attributes0);
      element0.append("`7.U<-Kbn,e!a2");
      assertEquals(1, element0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      CDataNode cDataNode0 = new CDataNode("Gf7dGCnV8H|y32");
      Attributes attributes0 = cDataNode0.attributes();
      ParseSettings parseSettings0 = htmlTreeBuilder0.defaultSettings();
      Tag tag0 = Tag.valueOf("tr", parseSettings0);
      Element element0 = new Element(tag0, "$*W)\"LuE+Mh<_", attributes0);
      element0.append("action");
      assertEquals(1, element0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      ParseSettings parseSettings0 = ParseSettings.htmlDefault;
      Attributes attributes0 = new Attributes();
      Tag tag0 = Tag.valueOf("tbody", parseSettings0);
      Element element0 = new Element(tag0, "address", attributes0);
      element0.append("org.jsoup.select.Evaluator$AttributeWithValue");
      assertEquals(1, element0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Parser.parse("listing", "tr");
      Element element0 = document0.head();
      Parser parser0 = Parser.xmlParser();
      List<Node> list0 = htmlTreeBuilder0.parseFragment("listing", element0, "I-/b_?w'~8B)5e/:ucO", parser0);
      assertEquals(0, element0.siblingIndex());
      assertEquals(0, element0.childNodeSize());
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      XmlTreeBuilder xmlTreeBuilder0 = new XmlTreeBuilder();
      Tag tag0 = Tag.valueOf("frameset");
      CDataNode cDataNode0 = new CDataNode("jp5dj?8_32_b");
      Attributes attributes0 = cDataNode0.attributes();
      Element element0 = new Element(tag0, "frameset", attributes0);
      Parser parser0 = new Parser(xmlTreeBuilder0);
      List<Node> list0 = htmlTreeBuilder0.parseFragment("zB{GY2", element0, "jp5dj?8_32_b", parser0);
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Tag tag0 = Tag.valueOf("html");
      Element element0 = new Element(tag0, "html", (Attributes) null);
      element0.append("|^-!#/N>|m0W;D,_0\"");
      assertEquals(2, element0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseInput("@Ct|T", "r/\">()#$  uo|'mym");
      htmlTreeBuilder0.insertStartTag("@Ct|T");
      htmlTreeBuilder0.resetInsertionMode();
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      Document document0 = parser0.parseInput("tr", "Gf7dGCnV8H|y32");
      assertEquals(1, document0.childNodeSize());
      
      boolean boolean0 = htmlTreeBuilder0.inTableScope("");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseInput("@Ct|T", "r/\">()#$  uo|'mym");
      Element element0 = htmlTreeBuilder0.insertStartTag("r/\">()#$  uo|'mym");
      assertEquals(1, element0.siblingIndex());
      
      boolean boolean0 = htmlTreeBuilder0.inTableScope("r/\">()#$  uo|'mym");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      Document document0 = parser0.parseInput("@Ct|T", "r\">()#$  o|'ym");
      assertEquals(1, document0.childNodeSize());
      
      boolean boolean0 = htmlTreeBuilder0.inListItemScope("r\">()#$  o|'ym");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseInput("@Ct|T", "dd");
      htmlTreeBuilder0.popStackToClose("YdW$");
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.inSelectScope("body");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Should not be reachable
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      Document document0 = parser0.parseInput("numeric reference with no numerals", "numeric reference with no numerals");
      assertEquals(1, document0.childNodeSize());
      
      boolean boolean0 = htmlTreeBuilder0.inSelectScope("numeric reference with no numerals");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      Document document0 = parser0.parseInput("numeric reference with no numerals", "numeric reference with no numerals");
      htmlTreeBuilder0.parseFragment("PKJm2wF7", document0, "W>F'IK$:}:U<*BV h}", parser0);
      assertEquals(1, document0.childNodeSize());
      
      boolean boolean0 = htmlTreeBuilder0.inSelectScope("html");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      ParseSettings parseSettings0 = ParseSettings.htmlDefault;
      Tag tag0 = Tag.valueOf("address", parseSettings0);
      CDataNode cDataNode0 = new CDataNode("address");
      Attributes attributes0 = cDataNode0.attributes();
      Element element0 = new Element(tag0, "address", attributes0);
      Parser parser0 = new Parser(htmlTreeBuilder0);
      htmlTreeBuilder0.parseFragment("address", element0, ",9,1]qrVR0q.jj", parser0);
      htmlTreeBuilder0.generateImpliedEndTags("html");
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseInput("@Ct|T", "@Ct|T");
      htmlTreeBuilder0.generateImpliedEndTags("@Ct|T");
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseInput("tr", "Gf7dGCnV8H|y32");
      Element element0 = htmlTreeBuilder0.removeLastFormattingElement();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      Document document0 = parser0.parseInput("&amp;", "&amp;");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.clearFormattingElementsToLastMarker();
      assertEquals(100, HtmlTreeBuilder.MaxScopeSearchDepth);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Attributes attributes0 = new Attributes();
      Tag tag0 = Tag.valueOf("table");
      Element element0 = new Element(tag0, "table", attributes0);
      Parser parser0 = new Parser(htmlTreeBuilder0);
      parser0.parseInput("&amp;", "&amp;");
      htmlTreeBuilder0.pushActiveFormattingElements(element0);
      htmlTreeBuilder0.removeFromActiveFormattingElements(element0);
      assertEquals("table", element0.tagName());
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      Document document0 = parser0.parseInput("@Ct|T", "dd");
      Element element0 = document0.head();
      htmlTreeBuilder0.pushActiveFormattingElements(element0);
      assertEquals(0, element0.childNodeSize());
      
      htmlTreeBuilder0.removeFromActiveFormattingElements(document0);
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = Parser.htmlParser();
      htmlTreeBuilder0.parseFragment("Th|R2>GJQb'", (Element) null, "n v+5_'`gpoj)p'!~p!", parser0);
      htmlTreeBuilder0.pushActiveFormattingElements((Element) null);
      Element element0 = htmlTreeBuilder0.getActiveFormattingElement("listing");
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Parser parser0 = new Parser(htmlTreeBuilder0);
      Document document0 = parser0.parseInput("@Ct|T", "dd");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      Element element0 = htmlTreeBuilder0.getActiveFormattingElement("_/+`pi;Q@xS");
      assertNull(element0);
  }
}
