/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:08:49 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.LinkedHashSet;
import java.util.LinkedList;
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
import org.jsoup.nodes.FormElement;
import org.jsoup.nodes.Node;
import org.jsoup.nodes.TextNode;
import org.jsoup.nodes.XmlDeclaration;
import org.jsoup.parser.Tag;
import org.jsoup.select.Elements;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Element_ESTest extends Element_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Document document0 = new Document("br");
      Element element0 = document0.appendText("br");
      Element element1 = document0.appendElement("br");
      assertEquals(1, element1.siblingIndex());
      
      Elements elements0 = element0.getElementsContainingText("br");
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Document document0 = new Document("br");
      Document document1 = (Document)document0.prependChild(document0);
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      // Undeclared exception!
      try { 
        document1.outerHtmlTail((StringBuilder) null, 46, document_OutputSettings0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Node", e);
      }
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Document document0 = new Document("U");
      // Undeclared exception!
      try { 
        document0.child(1488);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 1488, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Document document0 = new Document("br");
      Element element0 = document0.appendElement("br");
      element0.after((Node) document0);
      document0.html();
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Document document0 = new Document("-x~JW[wufDa1J\"sd");
      Element element0 = document0.prependText("-x~JW[wufDa1J\"sd");
      assertSame(element0, document0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Document document0 = new Document("br");
      document0.tagName("br");
      Element element0 = document0.appendElement("br");
      element0.after((Node) document0);
      document0.html();
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Document document0 = new Document("Pattern syntax error: ");
      Map<String, String> map0 = document0.dataset();
      assertEquals(0, map0.size());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Document document0 = new Document("br");
      Element element0 = document0.addClass("br");
      assertFalse(element0.isBlock());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Document document0 = new Document("Tag name mustYnotWbe empty.");
      Elements elements0 = document0.getElementsByAttributeValue("Tag name mustYnotWbe empty.", "Tag name mustYnotWbe empty.");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Document document0 = new Document("br");
      Elements elements0 = document0.getElementsByAttributeValueStarting("br", "br");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Document document0 = new Document("'r");
      Element element0 = document0.appendElement("'r");
      document0.parentNode = (Node) element0;
      assertEquals(0, document0.parentNode.siblingIndex());
      
      String string0 = document0.cssSelector();
      assertEquals("'r > #root", string0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Document document0 = new Document("U");
      Element element0 = document0.prepend("U");
      assertEquals("#root", element0.tagName());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Document document0 = new Document("U");
      Elements elements0 = document0.getElementsMatchingOwnText("U");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Document document0 = new Document("ui|");
      // Undeclared exception!
      try { 
        document0.after("ui|");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Document document0 = new Document("U");
      Elements elements0 = document0.getElementsByIndexLessThan(1972);
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Document document0 = new Document("");
      // Undeclared exception!
      try { 
        document0.getElementsByAttributeStarting("");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // String must not be empty
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Document document0 = new Document("P`)y>xu-&W(_9sb}I");
      Elements elements0 = document0.getElementsByIndexEquals(780);
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Document document0 = new Document("T$5)=W7FEATFi>1DT");
      // Undeclared exception!
      try { 
        document0.wrap("T$5)=W7FEATFi>1DT");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Node", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Document document0 = new Document("T>$'5)=W7FEAi>1DT");
      Element element0 = document0.val("T>$'5)=W7FEAi>1DT");
      assertEquals(0, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Document document0 = new Document("checked");
      Elements elements0 = document0.getElementsByAttributeValueMatching("checked", "checked");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Document document0 = new Document("'{2[alZ/eh|RA5Mj");
      Elements elements0 = document0.getElementsByAttributeValueEnding("'{2[alZ/eh|RA5Mj", "'{2[alZ/eh|RA5Mj");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Document document0 = new Document("-x~JW[wufDa1J\"sd");
      Elements elements0 = document0.getElementsByClass("-x~JW[wufDa1J\"sd");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Document document0 = new Document("br");
      Elements elements0 = document0.getElementsByAttributeValueContaining("br", "br");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Tag tag0 = Tag.valueOf("textarea");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "u", attributes0);
      formElement0.val("Tag name must not be empty.");
      assertEquals(1, formElement0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Document document0 = new Document("br");
      Document document1 = document0.clone();
      assertEquals(0, document1.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Document document0 = new Document("T$5)=W7FEATFi>1DT");
      Elements elements0 = document0.getElementsByAttributeValueNot("T$5)=W7FEATFi>1DT", "T$5)=W7FEATFi>1DT");
      assertFalse(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Document document0 = new Document("checke");
      String string0 = document0.toString();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Document document0 = new Document("checke");
      Elements elements0 = document0.getElementsByAttribute("checke");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Document document0 = new Document("U");
      // Undeclared exception!
      try { 
        document0.before("U");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Document document0 = new Document("TIg nme must not be empty.");
      Element element0 = document0.removeClass("TIg nme must not be empty.");
      assertFalse(element0.isBlock());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Document document0 = new Document("U");
      Elements elements0 = document0.getAllElements();
      assertFalse(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Document document0 = new Document("z0J!?A[b");
      Elements elements0 = document0.getElementsByTag("z0J!?A[b");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Document document0 = new Document("article");
      Elements elements0 = document0.getElementsByIndexGreaterThan((-1725691461));
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Document document0 = new Document("br");
      Element element0 = document0.appendElement("br");
      // Undeclared exception!
      try { 
        element0.html("br");
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Document document0 = new Document("br");
      Element element0 = document0.appendElement("br");
      document0.tagName("br");
      // Undeclared exception!
      try { 
        element0.html("br");
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Document document0 = new Document("br");
      document0.appendElement("br");
      List<TextNode> list0 = document0.textNodes();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Document document0 = new Document("U");
      Element element0 = document0.append("U");
      List<TextNode> list0 = element0.textNodes();
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Document document0 = new Document("org.jsoup.nodes.Ele-ent$1");
      DataNode dataNode0 = DataNode.createFromEncoded("(OPJ>-&'B-f0F[9", "p9yqW&e5&Fru&*{PI");
      document0.prependChild(dataNode0);
      List<DataNode> list0 = document0.dataNodes();
      assertTrue(list0.contains(dataNode0));
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Document document0 = new Document("U");
      Element element0 = document0.append("U");
      List<DataNode> list0 = element0.dataNodes();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Document document0 = new Document("T>$'5)=W7FEAi>1DT");
      LinkedList<FormElement> linkedList0 = new LinkedList<FormElement>();
      // Undeclared exception!
      try { 
        document0.insertChildren(38, linkedList0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Insert position out of bounds.
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Document document0 = new Document("br");
      LinkedList<DocumentType> linkedList0 = new LinkedList<DocumentType>();
      Element element0 = document0.insertChildren((-1), linkedList0);
      assertFalse(element0.isBlock());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Document document0 = new Document("b*ohufr");
      LinkedHashSet<XmlDeclaration> linkedHashSet0 = new LinkedHashSet<XmlDeclaration>();
      // Undeclared exception!
      try { 
        document0.insertChildren((-3889), linkedHashSet0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Insert position out of bounds.
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Document document0 = new Document("T$+5)=W7FEATFi>1DT");
      document0.toggleClass("T$+5)=W7FEATFi>1DT");
      String string0 = document0.cssSelector();
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Document document0 = new Document("T>$'5)=W7FEAi>1DT");
      Elements elements0 = document0.siblingElements();
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Document document0 = new Document("");
      Element element0 = document0.appendElement("#root");
      element0.before((Node) document0);
      Elements elements0 = element0.siblingElements();
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Document document0 = new Document("Y8;'-hB0WHFIheb|B");
      Element element0 = document0.appendElement("Y8;'-hB0WHFIheb|B");
      Element element1 = element0.nextElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Document document0 = new Document("java.lang.string@0000000028");
      Element element0 = document0.nextElementSibling();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Document document0 = new Document("Y8;'-hB0WHFIheb|B");
      Element element0 = document0.appendElement("Y8;'-hB0WHFIheb|B");
      element0.after((Node) document0);
      Document document1 = (Document)element0.nextElementSibling();
      assertEquals(Document.QuirksMode.noQuirks, document1.quirksMode());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Document document0 = new Document("Bq8Ejm5.?!CNk~v");
      Node[] nodeArray0 = new Node[6];
      nodeArray0[0] = (Node) document0;
      nodeArray0[1] = (Node) document0;
      TextNode textNode0 = new TextNode("Bq8Ejm5.?!CNk~v", "Bq8Ejm5.?!CNk~v");
      nodeArray0[2] = (Node) textNode0;
      nodeArray0[3] = (Node) document0;
      nodeArray0[4] = (Node) document0;
      nodeArray0[5] = (Node) document0;
      document0.addChildren(nodeArray0);
      Element element0 = document0.previousElementSibling();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Document document0 = new Document("T>$'5)=W7FEAi>1DT");
      Element element0 = document0.previousElementSibling();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Document document0 = new Document("Bq8Ejm5.?!CNk~v");
      Node[] nodeArray0 = new Node[6];
      Element element0 = document0.createElement("Bq8Ejm5.?!CNk~v");
      nodeArray0[0] = (Node) element0;
      nodeArray0[1] = (Node) document0;
      TextNode textNode0 = new TextNode("Bq8Ejm5.?!CNk~v", "Bq8Ejm5.?!CNk~v");
      nodeArray0[2] = (Node) textNode0;
      nodeArray0[3] = (Node) document0;
      nodeArray0[4] = (Node) document0;
      nodeArray0[5] = (Node) document0;
      document0.addChildren(nodeArray0);
      Element element1 = document0.previousElementSibling();
      assertFalse(element1.isBlock());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Document document0 = new Document("z}v!?R[b");
      Element element0 = document0.appendElement("z}v!?R[b");
      Element element1 = element0.firstElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Document document0 = new Document("T>$'5)=W7FEAi>1DT");
      Element element0 = document0.appendElement("#root");
      element0.before((Node) document0);
      Element element1 = element0.firstElementSibling();
      assertEquals(2, element1.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Document document0 = new Document("}!8x) q?");
      Element element0 = document0.appendElement("}!8x) q?");
      Integer integer0 = element0.elementSiblingIndex();
      assertEquals(0, (int)integer0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Document document0 = new Document("T$5)=W7FEATFi>1DT");
      Element element0 = document0.appendElement("T$5)=W7FEATFi>1DT");
      Element element1 = element0.lastElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Document document0 = new Document("z!?A[b");
      Element element0 = document0.appendElement("z!?A[b");
      element0.after((Node) document0);
      Element element1 = document0.lastElementSibling();
      assertEquals("#root", element1.tagName());
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Document document0 = new Document("article");
      Element element0 = document0.getElementById("article");
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Document document0 = new Document("summary");
      DataNode dataNode0 = new DataNode("summary", "");
      document0.appendChild(dataNode0);
      Elements elements0 = document0.getElementsMatchingText("summary");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Document document0 = new Document(">");
      Element element0 = document0.appendText(">");
      document0.appendElement(">");
      Elements elements0 = element0.getElementsContainingText(">");
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Document document0 = new Document("&$V");
      document0.append(",<!_2cf~AJW^");
      document0.getElementsContainingOwnText("&$V");
      assertEquals(2, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Document document0 = new Document("br");
      document0.appendElement("br");
      Elements elements0 = document0.getElementsContainingOwnText("br");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      Document document0 = new Document("t>$'5)=w7feai>1dt");
      document0.appendElement("t>$'5)=w7feai>1dt");
      Elements elements0 = document0.getElementsContainingOwnText("t>$'5)=w7feai>1dt");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      boolean boolean0 = Element.preserveWhitespace((Node) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      DataNode dataNode0 = DataNode.createFromEncoded("tu", "tu");
      boolean boolean0 = Element.preserveWhitespace(dataNode0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Document document0 = new Document("P`)y>xu-&W(_9sb}I");
      Element element0 = document0.appendElement("textarea");
      boolean boolean0 = Element.preserveWhitespace(element0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      Document document0 = new Document("br");
      Document document1 = new Document("br");
      document1.prependChild(document0);
      Element element0 = document0.appendText("br");
      Elements elements0 = element0.getElementsContainingText("br");
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      Document document0 = new Document("br");
      document0.appendElement("br");
      boolean boolean0 = document0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      Document document0 = new Document("co5'nv6$%b[xqgm}f");
      document0.appendText("");
      boolean boolean0 = document0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      Document document0 = new Document("br");
      DocumentType documentType0 = new DocumentType("br", "br", "s+", "br");
      document0.appendChild(documentType0);
      boolean boolean0 = document0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      Document document0 = new Document("\r~+Zp}G3DXt3");
      Element element0 = document0.appendElement("\r~+Zp}G3DXt3");
      element0.appendText("\r~+Zp}G3DXt3");
      boolean boolean0 = document0.hasText();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      Document document0 = new Document("bB");
      DataNode dataNode0 = DataNode.createFromEncoded("YoYXLZ?n", "CommentStart");
      document0.prependChild(dataNode0);
      String string0 = document0.data();
      assertEquals("YoYXLZ?n", string0);
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      Document document0 = new Document("bB");
      document0.append("bB");
      String string0 = document0.data();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      Document document0 = new Document("\r~+Zp}G3DXt3");
      document0.appendElement("\r~+Zp}G3DXt3");
      String string0 = document0.data();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test73()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Tag tag0 = Tag.valueOf("org.jsoup.parser.TreeBuilder");
      FormElement formElement0 = new FormElement(tag0, "", attributes0);
      formElement0.toggleClass("l|k-'34ea");
      boolean boolean0 = formElement0.hasClass("org.jsoup.parser.TreeBuilder");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test74()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Tag tag0 = Tag.valueOf("StartTag");
      FormElement formElement0 = new FormElement(tag0, "StartTag", attributes0);
      formElement0.toggleClass("StartTag");
      boolean boolean0 = formElement0.hasClass("StartTag");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test75()  throws Throwable  {
      Attributes attributes0 = new Attributes();
      Tag tag0 = Tag.valueOf("T}6'5)=W7FxAi>1DT");
      FormElement formElement0 = new FormElement(tag0, "T}6'5)=W7FxAi>1DT", attributes0);
      FormElement formElement1 = new FormElement(tag0, "T}6'5)=W7FxAi>1DT", attributes0);
      formElement0.toggleClass("T}6'5)=W7FxAi>1DT");
      boolean boolean0 = formElement0.hasClass("java.lang.string@0000000007");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test76()  throws Throwable  {
      Document document0 = new Document("T>$'5)=W7FEAi>1DT");
      String string0 = document0.val();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test77()  throws Throwable  {
      Document document0 = new Document("#root.java.lang.String@0000000007");
      Element element0 = document0.appendElement("textarea");
      String string0 = element0.val();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test78()  throws Throwable  {
      Document document0 = new Document("br");
      StringBuilder stringBuilder0 = new StringBuilder();
      document0.outerHtml(stringBuilder0);
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.prettyPrint(false);
      document0.outerHtmlHead(stringBuilder0, (-1842001214), document_OutputSettings1);
      assertEquals("<#root></#root><#root>", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test79()  throws Throwable  {
      Tag tag0 = Tag.valueOf("Q");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "Q", attributes0);
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      StringBuilder stringBuilder0 = new StringBuilder("Q");
      formElement0.outerHtmlHead(stringBuilder0, 908, document_OutputSettings0);
      assertEquals("Q<q>", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test80()  throws Throwable  {
      Tag tag0 = Tag.valueOf("Q");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "Q", attributes0);
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.outline(true);
      StringBuilder stringBuilder0 = new StringBuilder();
      stringBuilder0.append(1118);
      formElement0.outerHtmlHead(stringBuilder0, 31, document_OutputSettings1);
      assertEquals("1118\n                               <q>", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test81()  throws Throwable  {
      Document document0 = new Document("br");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings.Syntax document_OutputSettings_Syntax0 = Document.OutputSettings.Syntax.xml;
      document_OutputSettings0.syntax(document_OutputSettings_Syntax0);
      document0.outputSettings(document_OutputSettings0);
      document0.prependElement("br");
      String string0 = document0.html();
      assertEquals("<br />", string0);
  }

  @Test(timeout = 4000)
  public void test82()  throws Throwable  {
      Document document0 = new Document("t*p;9pv");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      document_OutputSettings0.prettyPrint(false);
      // Undeclared exception!
      try { 
        document0.outerHtmlTail((StringBuilder) null, 7, document_OutputSettings0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test83()  throws Throwable  {
      Document document0 = new Document("br");
      Document document1 = (Document)document0.prependChild(document0);
      document0.tagName("br");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      // Undeclared exception!
      try { 
        document1.outerHtmlTail((StringBuilder) null, 46, document_OutputSettings0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test84()  throws Throwable  {
      Document document0 = new Document("br");
      Document document1 = (Document)document0.prependChild(document0);
      document0.tagName("br");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      document_OutputSettings0.outline(true);
      // Undeclared exception!
      try { 
        document1.outerHtmlTail((StringBuilder) null, 46, document_OutputSettings0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Node", e);
      }
  }

  @Test(timeout = 4000)
  public void test85()  throws Throwable  {
      Tag tag0 = Tag.valueOf("Q");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "Q", attributes0);
      FormElement formElement1 = (FormElement)formElement0.append("Q");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      document_OutputSettings0.outline(true);
      formElement0.appendChild(formElement1);
      // Undeclared exception!
      try { 
        formElement1.outerHtmlTail((StringBuilder) null, 1, document_OutputSettings0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Node", e);
      }
  }

  @Test(timeout = 4000)
  public void test86()  throws Throwable  {
      Tag tag0 = Tag.valueOf("Q");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "Q", attributes0);
      FormElement formElement1 = (FormElement)formElement0.append("Q");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      document_OutputSettings0.outline(true);
      // Undeclared exception!
      try { 
        formElement1.outerHtmlTail((StringBuilder) null, 1, document_OutputSettings0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test87()  throws Throwable  {
      Document document0 = new Document("BB,#rQN/kKE");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.prettyPrint(false);
      Document document1 = document0.outputSettings(document_OutputSettings1);
      String string0 = document1.html();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test88()  throws Throwable  {
      Document document0 = new Document("#root.java.lang.String@0000000007");
      boolean boolean0 = document0.equals((Object) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test89()  throws Throwable  {
      Document document0 = new Document("br");
      Document document1 = new Document("x");
      boolean boolean0 = document0.equals(document1);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test90()  throws Throwable  {
      Document document0 = new Document("T>$'5)=W7FEAi>1DT");
      Document document1 = (Document)document0.doClone(document0);
      document0.prependChild(document0);
      // Undeclared exception!
      try { 
        document1.nextElementSibling();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test91()  throws Throwable  {
      Tag tag0 = Tag.valueOf("Q");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "Q", attributes0);
      formElement0.hashCode();
  }
}
