/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 15:11:45 GMT 2023
 */

package org.jsoup.parser;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.LinkedList;
import java.util.List;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Attributes;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.FormElement;
import org.jsoup.nodes.Node;
import org.jsoup.nodes.TextNode;
import org.jsoup.parser.CharacterReader;
import org.jsoup.parser.HtmlTreeBuilder;
import org.jsoup.parser.ParseErrorList;
import org.jsoup.parser.Tag;
import org.jsoup.parser.Token;
import org.jsoup.parser.Tokeniser;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class HtmlTreeBuilder_ESTest extends HtmlTreeBuilder_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("le@C>/I\";IeK", "le@C>/I\";IeK");
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
      Element element0 = htmlTreeBuilder0.getHeadElement();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("0P[/PM'DBiekWX_", "0P[/PM'DBiekWX_");
      CharacterReader characterReader0 = new CharacterReader("0P[/PM'DBiekWX_");
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      Tokeniser tokeniser0 = new Tokeniser(characterReader0, parseErrorList0);
      Token.StartTag token_StartTag0 = tokeniser0.startPending;
      Attributes attributes0 = new Attributes();
      Token.StartTag token_StartTag1 = token_StartTag0.nameAttr("noscript", attributes0);
      htmlTreeBuilder0.insertForm(token_StartTag1, true);
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("DoctypePublicIdentifier_doubleQuoted", "DoctypePublicIdentifier_doubleQuoted");
      assertEquals(1, document0.childNodeSize());
      
      boolean boolean0 = htmlTreeBuilder0.processEndTag("tf");
      assertFalse(boolean0);
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
         verifyException("org.jsoup.parser.TreeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("tfoot", "tfoot");
      Element element0 = document0.prependElement("tfoot");
      ParseErrorList parseErrorList0 = htmlTreeBuilder0.errors;
      List<Node> list0 = htmlTreeBuilder0.parseFragment(";:luX^a?NhZ-vT", element0, "rt", parseErrorList0);
      assertEquals(2, document0.childNodeSize());
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.insertMarkerToFormattingElements();
      htmlTreeBuilder0.clearFormattingElementsToLastMarker();
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
        htmlTreeBuilder0.inButtonScope("param");
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
      htmlTreeBuilder0.parse("select", "select");
      htmlTreeBuilder0.processStartTag("select");
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      LinkedList<String> linkedList0 = new LinkedList<String>();
      htmlTreeBuilder0.setPendingTableCharacters(linkedList0);
      assertEquals(0, linkedList0.size());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.getDocument();
      htmlTreeBuilder0.pushActiveFormattingElements((Element) null);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.inScope("V?$E`O6");
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
      Document document0 = htmlTreeBuilder0.parse("d[iar|=c^^JD^)}8)", "d[iar|=c^^JD^)}8)");
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
  public void test18()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.setFosterInserts(true);
      Document document0 = htmlTreeBuilder0.parse("wr", "wr");
      assertEquals(1, document0.childNodeSize());
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
      Document document0 = htmlTreeBuilder0.parse("P*NEpB[bH<![fw6trR", "P*NEpB[bH<![fw6trR");
      assertEquals(1, document0.childNodeSize());
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
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      List<Node> list0 = htmlTreeBuilder0.parseFragment("DZ.0?V5JqbBj4^^y> ", (Element) null, "DZ.0?V5JqbBj4^^y> ", parseErrorList0);
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Tag tag0 = Tag.valueOf("frameset");
      Attributes attributes0 = new Attributes();
      Element element0 = new Element(tag0, "org.Rsoup.parser.PareEQror", attributes0);
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      List<Node> list0 = htmlTreeBuilder0.parseFragment("org.Rsoup.parser.PareEQror", element0, "org.Rsoup.parser.PareEQror", parseErrorList0);
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("select", "select");
      Element element0 = document0.createElement("title");
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      List<Node> list0 = htmlTreeBuilder0.parseFragment("title", element0, "select", parseErrorList0);
      assertEquals(1, list0.size());
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("style", "style");
      ParseErrorList parseErrorList0 = ParseErrorList.tracking(1372);
      Element element0 = htmlTreeBuilder0.insertStartTag("style");
      List<Node> list0 = htmlTreeBuilder0.parseFragment("mwG^QfiJd", element0, "mwG^QfiJd", parseErrorList0);
      assertTrue(parseErrorList0.isEmpty());
      assertEquals(1, list0.size());
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("DoctypeSystemIdentifier_singleQuoted", "DoctypeSystemIdentifier_singleQuoted");
      Element element0 = document0.prependElement("script");
      ParseErrorList parseErrorList0 = new ParseErrorList(13, 13);
      htmlTreeBuilder0.parseFragment("DoctypeSystemIdentifier_singleQuoted", element0, "DoctypeSystemIdentifier_singleQuoted", parseErrorList0);
      assertEquals(2, document0.childNodeSize());
      assertTrue(parseErrorList0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("noscript", "noscript");
      Attributes attributes0 = new Attributes();
      Tokeniser tokeniser0 = htmlTreeBuilder0.tokeniser;
      Token.StartTag token_StartTag0 = tokeniser0.startPending;
      Token.StartTag token_StartTag1 = token_StartTag0.nameAttr("noscript", attributes0);
      Element element0 = htmlTreeBuilder0.insertEmpty(token_StartTag1);
      ParseErrorList parseErrorList0 = htmlTreeBuilder0.errors;
      List<Node> list0 = htmlTreeBuilder0.parseFragment("noscript", element0, "^\"P^;`+v\u0007Zg!c~P>;=", parseErrorList0);
      assertEquals(1, element0.siblingIndex());
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = new Document("PW,9P,U0bUQ}52#I");
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      Element element0 = document0.prependElement("plaintext");
      List<Node> list0 = htmlTreeBuilder0.parseFragment("plaintext", element0, "plaintext", parseErrorList0);
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Tag tag0 = Tag.valueOf("dq\"E?B%kmXno'");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "l,$.", attributes0);
      List<Node> list0 = htmlTreeBuilder0.parseFragment("dq\"E?B%kmXno'", formElement0, "#root", (ParseErrorList) null);
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Document.createShell("org.jsoup.helper.W3CDom$W3CBuilder");
      htmlTreeBuilder0.maybeSetBaseUri(document0);
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("Docty6eSy}temIdentifier_singleQuoted", "Docty6eSy}temIdentifier_singleQuoted");
      ParseErrorList parseErrorList0 = new ParseErrorList(1517, 3240);
      List<Node> list0 = htmlTreeBuilder0.parseFragment("Docty6eSy}temIdentifier_singleQuoted", document0, "Docty6eSy}temIdentifier_singleQuoted", parseErrorList0);
      assertEquals(1, list0.size());
      
      boolean boolean0 = htmlTreeBuilder0.processStartTag("th");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
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
  public void test33()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("&amp;", "noscript");
      Token.StartTag token_StartTag0 = new Token.StartTag();
      Attributes attributes0 = new Attributes();
      Token.StartTag token_StartTag1 = token_StartTag0.nameAttr("td", attributes0);
      FormElement formElement0 = htmlTreeBuilder0.insertForm(token_StartTag1, false);
      assertEquals("noscript", formElement0.baseUri());
      assertEquals(1, formElement0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("u2XZ=EZ^jW-/[", "u2XZ=EZ^jW-/[");
      htmlTreeBuilder0.processStartTag("script");
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
  public void test35()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("-Qr)P8U6]T", "-Qr)P8U6]T");
      htmlTreeBuilder0.processStartTag("style");
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
      Tag tag0 = Tag.valueOf("tr");
      TextNode textNode0 = new TextNode("tr", "tr");
      Attributes attributes0 = textNode0.attributes();
      FormElement formElement0 = new FormElement(tag0, "tr", attributes0);
      htmlTreeBuilder0.setFormElement(formElement0);
      htmlTreeBuilder0.parse("select", "select");
      boolean boolean0 = htmlTreeBuilder0.processStartTag("select");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("le@C>/I\";IeK", "le@C>/I\";IeK");
      Element element0 = htmlTreeBuilder0.currentElement();
      Element element1 = htmlTreeBuilder0.aboveOnStack(element0);
      List<Node> list0 = htmlTreeBuilder0.parseFragment("le@C>/I\";IeK", element1, "le@C>/I\";IeK", (ParseErrorList) null);
      assertEquals(2, list0.size());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("span", "}1$CRi7'hmIW");
      htmlTreeBuilder0.insertStartTag("form");
      Element element0 = htmlTreeBuilder0.getFromStack("form");
      assertEquals("}1$CRi7'hmIW", element0.baseUri());
      assertEquals(1, element0.siblingIndex());
      assertNotNull(element0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("-Qr)P8U6]T", "-Qr)P8U6]T");
      boolean boolean0 = htmlTreeBuilder0.removeFromStack(document0);
      assertFalse(boolean0);
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("param", "_6~-h");
      Element element0 = document0.body();
      boolean boolean0 = htmlTreeBuilder0.removeFromStack(element0);
      assertTrue(boolean0);
      assertEquals(1, element0.siblingIndex());
      assertEquals("_6~-h", element0.baseUri());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("DoctypePublicIdentifier_doubleQuoted", "DoctypePublicIdentifier_doubleQuoted");
      boolean boolean0 = htmlTreeBuilder0.processStartTag("tf");
      boolean boolean1 = htmlTreeBuilder0.processEndTag("tf");
      assertTrue(boolean1 == boolean0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("DoctypeSy}temIdentifier_singleQuoted", "DoctypeSy}temIdentifier_singleQuoted");
      String[] stringArray0 = new String[0];
      htmlTreeBuilder0.popStackToClose(stringArray0);
      assertEquals(0, stringArray0.length);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("DoctypeSystemIdentifier_singleQuoted", "isindex");
      htmlTreeBuilder0.popStackToClose(htmlTreeBuilder0.TagsSearchInScope);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.initialiseParse("article", "article", (ParseErrorList) null);
      htmlTreeBuilder0.clearStackToTableBodyContext();
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("style", "style");
      htmlTreeBuilder0.clearStackToTableBodyContext();
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("wMk(Dq4", "wMk(Dq4");
      Element element0 = htmlTreeBuilder0.currentElement();
      Element element1 = htmlTreeBuilder0.aboveOnStack(element0);
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.aboveOnStack(element1);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");
      
      } catch(ArrayIndexOutOfBoundsException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("hw~.7qpv}by+ppd?tn#", "hw~.7qpv}by+ppd?tn#", (ParseErrorList) null);
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
  public void test48()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("(S5ho3", "(S5ho3", (ParseErrorList) null);
      Element element0 = document0.body();
      htmlTreeBuilder0.insertOnStackAfter(element0, element0);
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("}", "=[[X7oE*,");
      Element element0 = htmlTreeBuilder0.currentElement();
      htmlTreeBuilder0.replaceOnStack(element0, element0);
      assertEquals("=[[X7oE*,", element0.baseUri());
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("isindex", "isindex");
      htmlTreeBuilder0.popStackToBefore("isindex");
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("0P[/PM'DBiekWX_", "0P[/PM'DBiekWX_");
      htmlTreeBuilder0.resetInsertionMode();
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("^[", "d[iar)=U^^Jq^)}8)");
      Element element0 = document0.createElement("td");
      List<Node> list0 = htmlTreeBuilder0.parseFragment("Z%mh)dD)Ph,>P@", element0, "^[", (ParseErrorList) null);
      assertEquals("d[iar)=U^^Jq^)}8)", element0.baseUri());
      assertEquals(1, document0.childNodeSize());
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Document.createShell("tr");
      Element element0 = document0.prependElement("tr");
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.parseFragment("optgroup", element0, "RcdataLessthanSign", (ParseErrorList) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.HtmlTreeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("td", "org.jsoup.nodes.Attribute", (ParseErrorList) null);
      Element element0 = document0.prependElement("tbody");
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.parseFragment("div", element0, "td", (ParseErrorList) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.HtmlTreeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Document.createShell("Y");
      Element element0 = document0.prependElement("thead");
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.parseFragment("href", element0, "WWx%D", (ParseErrorList) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.HtmlTreeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("caption", "rc");
      Element element0 = document0.prependElement("caption");
      htmlTreeBuilder0.parseFragment("caption", element0, "caption", (ParseErrorList) null);
      assertEquals(2, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("org.jsoup.parser.HtmlTreeBuilder", "K-RfPP");
      Element element0 = document0.prependElement("colgroup");
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.parseFragment("K-RfPP", element0, "Jw", (ParseErrorList) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.HtmlTreeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("K,D&8BU", "K,D&8BU", (ParseErrorList) null);
      Element element0 = document0.prependElement("table");
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.parseFragment("table", element0, "K,D&8BU", (ParseErrorList) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.parser.HtmlTreeBuilder", e);
      }
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("OnO{wK6c5l", "OnO{wK6c5l");
      Element element0 = document0.head();
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      htmlTreeBuilder0.parseFragment("OnO{wK6c5l", element0, "OnO{wK6c5l", parseErrorList0);
      assertEquals(0, element0.childNodeSize());
      assertEquals(0, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.initialiseParse("tfoot", "tfoot", (ParseErrorList) null);
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.inTableScope("tfoot");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Should not be reachable
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("&", "&");
      assertEquals(1, document0.childNodeSize());
      
      boolean boolean0 = htmlTreeBuilder0.inScope(htmlTreeBuilder0.TagsSearchInScope);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("param", "param");
      assertEquals(1, document0.childNodeSize());
      
      boolean boolean0 = htmlTreeBuilder0.inTableScope("param");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("ol", "ol");
      assertEquals(1, document0.childNodeSize());
      
      boolean boolean0 = htmlTreeBuilder0.inListItemScope("ol");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("form", "form");
      htmlTreeBuilder0.popStackToClose("form");
      // Undeclared exception!
      try { 
        htmlTreeBuilder0.inSelectScope("command");
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
      Document document0 = htmlTreeBuilder0.parse("sc>ipt", "sc>ipt");
      assertEquals(1, document0.childNodeSize());
      
      boolean boolean0 = htmlTreeBuilder0.inSelectScope("sc>ipt");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("d[ar)=U^^Jq^}8)", "d[ar)=U^^Jq^}8)");
      assertEquals(1, document0.childNodeSize());
      
      boolean boolean0 = htmlTreeBuilder0.inSelectScope("body");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("style", "style");
      htmlTreeBuilder0.generateImpliedEndTags("mwG^QfiJd");
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Document.createShell("org.jsoup.helper.W3CDom$W3CBuilder");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      htmlTreeBuilder0.parseFragment("org.jsoup.helper.W3CDom$W3CBuilder", document0, "org.jsoup.helper.W3CDom$W3CBuilder", parseErrorList0);
      htmlTreeBuilder0.removeFromActiveFormattingElements(document0);
      assertFalse(document0.updateMetaCharsetElement());
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Element element0 = htmlTreeBuilder0.removeLastFormattingElement();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("org.jsoup.helper.W3CDom$W3CBuilder", "org.jsoup.helper.W3CDom$W3CBuilder", (ParseErrorList) null);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.clearFormattingElementsToLastMarker();
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Document.createShell("me^>bGjD[");
      Element element0 = document0.prependElement("me^>bGjD[");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.pushActiveFormattingElements(element0);
      assertEquals(0, element0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("Unexpected token [%s] when in state [%s]", "Unexpected token [%s] when in state [%s]");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test73()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("Unexpected token [%s] when in state [%s]", "Unexpected token [%s] when in state [%s]");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.parse("Unexpected token [%s] when in state [%s]", "*gCqtH*ULd=K0*,l\"y");
      boolean boolean0 = htmlTreeBuilder0.processStartTag("Unexpected token [%s] when in state [%s]");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test74()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("-Qr)P8U6]T", "-Qr)P8U6]T");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      Document document1 = htmlTreeBuilder0.parse("-Qr)P8U6]T", "-Qr)P8U6]T");
      assertFalse(document1.equals((Object)document0));
      assertEquals(1, document1.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test75()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.pushActiveFormattingElements((Element) null);
      ParseErrorList parseErrorList0 = ParseErrorList.noTracking();
      Document document0 = htmlTreeBuilder0.parse("doctypesy}temidentifier_singlequoted", "X}imNTl3A8.Qe\";!Rg", parseErrorList0);
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      Document document1 = htmlTreeBuilder0.parse("UoSLj^]", "j");
      assertFalse(document1.equals((Object)document0));
  }

  @Test(timeout = 4000)
  public void test76()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("7]rvFducNLlr7", "|;|C");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      Document document1 = htmlTreeBuilder0.parse("7]rvFducNLlr7", "|;|C");
      htmlTreeBuilder0.pushActiveFormattingElements(document1);
      htmlTreeBuilder0.reconstructFormattingElements();
  }

  @Test(timeout = 4000)
  public void test77()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Document.createShell("org.jsoup.helper.W3CDom$W3CBuilder");
      htmlTreeBuilder0.removeFromActiveFormattingElements(document0);
      assertFalse(document0.updateMetaCharsetElement());
  }

  @Test(timeout = 4000)
  public void test78()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = Document.createShell("org.jsoup.helper.W3CDom$W3CBuilder");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      htmlTreeBuilder0.removeFromActiveFormattingElements(document0);
      assertEquals(Document.QuirksMode.noQuirks, document0.quirksMode());
  }

  @Test(timeout = 4000)
  public void test79()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      Document document0 = htmlTreeBuilder0.parse("DoctypeSystemIdentifier_singleQuoted", "DoctypeSystemIdentifier_singleQuoted");
      htmlTreeBuilder0.pushActiveFormattingElements(document0);
      Element element0 = htmlTreeBuilder0.getActiveFormattingElement("async");
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test80()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.pushActiveFormattingElements((Element) null);
      Element element0 = htmlTreeBuilder0.getActiveFormattingElement("h96M1^C4}^");
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test81()  throws Throwable  {
      HtmlTreeBuilder htmlTreeBuilder0 = new HtmlTreeBuilder();
      htmlTreeBuilder0.parse("DoctypeSystemIdentifier_singleQuoted", "DoctypeSystemIdentifier_singleQuoted");
      Element element0 = htmlTreeBuilder0.insertStartTag("async");
      htmlTreeBuilder0.pushActiveFormattingElements(element0);
      Element element1 = htmlTreeBuilder0.getActiveFormattingElement("async");
      assertEquals(1, element1.siblingIndex());
      assertNotNull(element1);
  }
}
