/*
 * This file was automatically generated by EvoSuite
 * Tue Sep 26 17:48:45 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Attributes;
import org.jsoup.nodes.DataNode;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.DocumentType;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Node;
import org.jsoup.nodes.TextNode;
import org.jsoup.parser.Tag;
import org.jsoup.select.Elements;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Element_ESTest extends Element_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Document document0 = new Document("jSscA");
      Document document1 = (Document)document0.prepend("jSscA");
      StringBuilder stringBuilder0 = new StringBuilder();
      document1.outerHtml(stringBuilder0);
      assertEquals("<#root>\n jSscA\n</#root>", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Document document0 = new Document("because");
      Elements elements0 = document0.getElementsMatchingText("because");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Document document0 = new Document("i4h9zdwz9opzu+:");
      // Undeclared exception!
      try { 
        document0.child(819);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 819, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Document document0 = new Document("Patt.rn syNtex\"error: ");
      Element element0 = document0.prependText("Patt.rn syNtex\"error: ");
      assertEquals(0, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Document document0 = new Document("0h]dLWmw$");
      Document document1 = (Document)document0.tagName("0h]dLWmw$");
      assertEquals(Document.QuirksMode.noQuirks, document1.quirksMode());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Tag tag0 = Tag.valueOf("textarea");
      Attributes attributes0 = new Attributes();
      Element element0 = new Element(tag0, "ojw<ehZR", attributes0);
      Element element1 = element0.val("");
      assertSame(element0, element1);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Document document0 = new Document("SscA");
      Map<String, String> map0 = document0.dataset();
      assertTrue(map0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Document document0 = new Document("0h]dWmw$");
      Element element0 = document0.addClass("0h]dWmw$");
      assertEquals(0, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Document document0 = new Document("Pattern syntax erroE: ");
      // Undeclared exception!
      try { 
        document0.before((Node) document0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Document document0 = new Document("because");
      // Undeclared exception!
      try { 
        document0.html("because");
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Document document0 = new Document("because");
      String string0 = "yw~{s\\uu";
      Elements elements0 = document0.getElementsByAttributeValue(string0, "because");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Document document0 = new Document("0h]dlwmw$");
      // Undeclared exception!
      try { 
        document0.siblingElements();
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Element", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Document document0 = new Document("`#92~");
      Elements elements0 = document0.getElementsByAttributeValueStarting("`#92~", "`#92~");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Document document0 = new Document("0h]dlwmw$");
      // Undeclared exception!
      try { 
        document0.select("0h]dlwmw$");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Could not parse query '0h]dlwmw$': unexpected token at ']dlwmw$'
         //
         verifyException("org.jsoup.select.QueryParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Document document0 = new Document("becaus-w");
      // Undeclared exception!
      try { 
        document0.after((String) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Document document0 = new Document(",M");
      document0.prependElement("}=");
      Element element0 = document0.prependChild(document0);
      Element element1 = element0.nextElementSibling();
      assertEquals("}=", element1.nodeName());
      assertNotNull(element1);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Document document0 = new Document(",M");
      Elements elements0 = document0.getElementsByAttributeStarting(",M");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Document document0 = new Document("id");
      Elements elements0 = document0.getElementsByIndexEquals(18);
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Document document0 = new Document("");
      // Undeclared exception!
      try { 
        document0.wrap("");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // String must not be empty
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Document document0 = new Document("aopf");
      Element element0 = document0.val("text");
      assertEquals("#document", element0.nodeName());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Document document0 = new Document(",~");
      String string0 = document0.toString();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Document document0 = new Document("0h]dlwmw$");
      Elements elements0 = document0.getElementsByAttributeValueMatching("0h]dlwmw$", "0h]dlwmw$");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Document document0 = new Document("foo!%j8c0f-=");
      Elements elements0 = document0.getElementsByAttributeValueEnding("foo!%j8c0f-=", "vfr");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Document document0 = new Document("^#4HUCH{>R");
      Elements elements0 = document0.getElementsByClass("%/LZ\"c<N2bTE$,9)");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Document document0 = new Document("becaus-w");
      Elements elements0 = document0.getElementsContainingText("becaus-w");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Document document0 = new Document(">");
      Elements elements0 = document0.getElementsByAttributeValueContaining("|", ">");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Document document0 = new Document("0h]dlwmw$");
      // Undeclared exception!
      try { 
        document0.after((Node) document0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Document document0 = new Document("Pattern syntax error: ");
      Document document1 = document0.clone();
      Document document2 = document1.clone();
      assertNotSame(document2, document0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Document document0 = new Document("br");
      Elements elements0 = document0.getElementsByAttributeValueNot("br", "br");
      assertFalse(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Document document0 = new Document("0h]dWmw$");
      Elements elements0 = document0.getElementsByAttribute("0h]dWmw$");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Document document0 = new Document("because");
      // Undeclared exception!
      try { 
        document0.before("because");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Document document0 = new Document(" ");
      // Undeclared exception!
      try { 
        document0.normalise();
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Document document0 = new Document("becausw");
      Element element0 = document0.removeClass("becausw");
      assertSame(document0, element0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Document document0 = new Document("0h]dLWmw$");
      Elements elements0 = document0.getAllElements();
      assertFalse(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Document document0 = new Document(",M");
      String string0 = document0.title();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Document document0 = new Document("|0i9my:zWOJY4KK9*");
      Elements elements0 = document0.getElementsByIndexGreaterThan((-706));
      assertFalse(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Document document0 = new Document(",~");
      document0.prependChild(document0);
      Elements elements0 = document0.parents();
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Tag tag0 = Tag.valueOf("text");
      Attributes attributes0 = new Attributes();
      Element element0 = new Element(tag0, "", attributes0);
      Document document0 = new Document("aopf");
      element0.prependChild(document0);
      Elements elements0 = document0.parents();
      assertFalse(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Document document0 = new Document("becausw");
      document0.prependChild(document0);
      Element element0 = document0.firstElementSibling();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Document document0 = new Document("jSscA");
      Element element0 = document0.prepend("jSscA");
      Elements elements0 = element0.children();
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Tag tag0 = Tag.valueOf("o7w<e,ZR");
      Element element0 = new Element(tag0, "o7w<e,ZR");
      element0.text("o7w<e,ZR");
      List<TextNode> list0 = element0.textNodes();
      assertFalse(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      String string0 = "Yw~{S\\uu";
      Tag tag0 = Tag.valueOf(string0);
      Attributes attributes0 = new Attributes();
      Element element0 = new Element(tag0, "ojw<ehZR", attributes0);
      Document document0 = new Document("#NqWb~");
      element0.prependChild(document0);
      List<TextNode> list0 = element0.textNodes();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Document document0 = new Document("SscA");
      document0.prepend("SscA");
      List<DataNode> list0 = document0.dataNodes();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Document document0 = new Document(",M");
      Element element0 = document0.prependChild(document0);
      Element element1 = element0.nextElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Tag tag0 = Tag.valueOf("o7w<e,ZR");
      Attributes attributes0 = new Attributes();
      Element element0 = new Element(tag0, "o7w<e,ZR", attributes0);
      element0.prependChild(element0);
      Element element1 = element0.previousElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Tag tag0 = Tag.valueOf("text");
      Attributes attributes0 = new Attributes();
      Element element0 = new Element(tag0, "", attributes0);
      Document document0 = new Document("aopf");
      Element element1 = element0.prependChild(document0);
      element0.prependChild(element1);
      Element element2 = document0.previousElementSibling();
      assertEquals("text", element2.nodeName());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Document document0 = new Document(",M");
      Document document1 = document0.clone();
      Element element0 = document0.prependChild(document1);
      document0.prependChild(element0);
      assertEquals(1, document1.siblingIndex());
      
      document0.firstElementSibling();
      assertNotSame(document0, document1);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Document document0 = new Document(",M");
      Element element0 = document0.prependChild(document0);
      element0.getElementsByIndexLessThan((-480));
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Tag tag0 = Tag.valueOf("o7w<e,ZR");
      Attributes attributes0 = new Attributes();
      Element element0 = new Element(tag0, "o7w<e,ZR", attributes0);
      Element element1 = element0.prependChild(element0);
      Element element2 = element1.lastElementSibling();
      assertNull(element2);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Tag tag0 = Tag.valueOf("text");
      Attributes attributes0 = new Attributes();
      Element element0 = new Element(tag0, "", attributes0);
      Document document0 = new Document("aopf");
      Element element1 = element0.prependChild(document0);
      element0.prependChild(element1);
      Element element2 = document0.lastElementSibling();
      assertFalse(element2.isBlock());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Tag tag0 = Tag.valueOf("o7w<e,ZR");
      Element element0 = new Element(tag0, "o7w<e,ZR");
      element0.setParentNode(element0);
      // Undeclared exception!
      try { 
        element0.nextElementSibling();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Document document0 = new Document("because");
      Element element0 = document0.getElementById("because");
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Document document0 = new Document("because");
      DataNode dataNode0 = new DataNode("because", "because");
      Element element0 = document0.appendChild(dataNode0);
      String string0 = element0.text();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Tag tag0 = Tag.valueOf("text");
      Attributes attributes0 = new Attributes();
      Element element0 = new Element(tag0, "", attributes0);
      Document document0 = new Document("lopf");
      Element element1 = element0.prependChild(document0);
      String string0 = element1.text();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Document document0 = new Document("id");
      DocumentType documentType0 = new DocumentType("E", "<", ",~", ",");
      document0.prependChild(documentType0);
      Elements elements0 = document0.getElementsMatchingOwnText(",");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Document document0 = new Document("0h]dN~Wmw$");
      Element element0 = document0.append("because");
      Elements elements0 = element0.getElementsContainingOwnText("character outside of valid range");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Document document0 = new Document("jscr");
      Element element0 = document0.prependChild(document0);
      // Undeclared exception!
      element0.getElementsContainingOwnText("jscr");
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Document document0 = new Document("Patt.rn syNtex\"error: ");
      Document document1 = new Document("Patt.rn syNtex\"error: ");
      Element element0 = document1.appendText("Patt.rn syNtex\"error: ");
      Element element1 = element0.prependChild(document0);
      element1.prepend("Patt.rn syNtex\"error: ");
      String string0 = document1.text();
      assertEquals("Patt.rn syNtex\"error: Patt.rn syNtex\"error:", string0);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Tag tag0 = Tag.valueOf("br");
      Element element0 = new Element(tag0, "textarea");
      Node[] nodeArray0 = new Node[3];
      nodeArray0[0] = (Node) element0;
      nodeArray0[1] = (Node) element0;
      nodeArray0[2] = (Node) element0;
      element0.addChildren(nodeArray0);
      // Undeclared exception!
      try { 
        element0.text();
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Document document0 = new Document("Pattern syntax error: ");
      Document document1 = new Document("Pattern syntax error: ");
      document1.prependChild(document0);
      document0.prepend("Pattern syntax error: ");
      String string0 = document0.text();
      assertEquals("Pattern syntax error:", string0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Document document0 = new Document("id");
      DocumentType documentType0 = new DocumentType("E", "<", ",~", ",");
      Element element0 = document0.prependChild(documentType0);
      boolean boolean0 = element0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      Document document0 = new Document("lopf");
      Element element0 = document0.appendText("");
      boolean boolean0 = element0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      Tag tag0 = Tag.valueOf(",~");
      Attributes attributes0 = new Attributes();
      Element element0 = new Element(tag0, ",~", attributes0);
      Document document0 = new Document(",~");
      document0.prependChild(element0);
      element0.appendText(",~");
      boolean boolean0 = document0.hasText();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      Tag tag0 = Tag.valueOf("text");
      Attributes attributes0 = new Attributes();
      Element element0 = new Element(tag0, "", attributes0);
      Document document0 = new Document("lopf");
      element0.prependChild(document0);
      boolean boolean0 = element0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Tag tag0 = Tag.valueOf("because");
      Element element0 = new Element(tag0, "because");
      element0.text("because");
      String string0 = element0.data();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      Document document0 = new Document(",M");
      document0.prependChild(document0);
      // Undeclared exception!
      try { 
        document0.data();
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Tag tag0 = Tag.valueOf(" />");
      Element element0 = new Element(tag0, " />", attributes0);
      boolean boolean0 = element0.hasClass("");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      Document document0 = new Document("*;Xs\"]*)#1cOreL");
      Element element0 = document0.toggleClass("*;Xs\"]*)#1cOreL");
      assertSame(document0, element0);
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      Document document0 = new Document("");
      Element element0 = document0.toggleClass("");
      assertEquals("#document", element0.nodeName());
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      Tag tag0 = Tag.valueOf(",~");
      Element element0 = new Element(tag0, ",~");
      String string0 = element0.val();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      Tag tag0 = Tag.valueOf("textarea");
      Attributes attributes0 = new Attributes();
      Element element0 = new Element(tag0, "ojw<ehZR", attributes0);
      String string0 = element0.val();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      Document document0 = new Document("^#4HUCH{>R");
      StringBuilder stringBuilder0 = new StringBuilder((CharSequence) "^#4HUCH{>R");
      document0.outerHtml(stringBuilder0);
      assertEquals("^#4HUCH{>R\n<#root></#root>", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      Document document0 = new Document("becausw");
      StringBuilder stringBuilder0 = new StringBuilder("becausw");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.prettyPrint(false);
      document0.outerHtmlHead(stringBuilder0, 660, document_OutputSettings1);
      assertEquals("becausw<#root>", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test73()  throws Throwable  {
      Document document0 = new Document(" ");
      StringBuilder stringBuilder0 = new StringBuilder(" ");
      Document.OutputSettings document_OutputSettings0 = document0.outputSettings();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.prettyPrint(false);
      document0.outerHtmlTail(stringBuilder0, 6, document_OutputSettings1);
      assertEquals(" </#root>", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test74()  throws Throwable  {
      Document document0 = new Document("id");
      DocumentType documentType0 = new DocumentType("E", "<", ",~", ",");
      Element element0 = document0.prependChild(documentType0);
      String string0 = element0.outerHtml();
      assertEquals("<!DOCTYPE E PUBLIC \"<\" \",~\">", string0);
  }

  @Test(timeout = 4000)
  public void test75()  throws Throwable  {
      Tag tag0 = Tag.valueOf("textarea");
      Attributes attributes0 = new Attributes();
      Element element0 = new Element(tag0, "ojw<ehZR", attributes0);
      element0.hashCode();
  }
}