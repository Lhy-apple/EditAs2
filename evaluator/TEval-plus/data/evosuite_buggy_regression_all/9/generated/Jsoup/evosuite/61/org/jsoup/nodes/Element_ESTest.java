/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:11:57 GMT 2023
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
import org.evosuite.runtime.mock.java.io.MockFileWriter;
import org.evosuite.runtime.mock.java.io.MockPrintStream;
import org.jsoup.nodes.Attributes;
import org.jsoup.nodes.Comment;
import org.jsoup.nodes.DataNode;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.DocumentType;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.FormElement;
import org.jsoup.nodes.Node;
import org.jsoup.nodes.TextNode;
import org.jsoup.parser.Tag;
import org.jsoup.select.Elements;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Element_ESTest extends Element_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Document document0 = new Document("col");
      Element element0 = document0.appendText("col");
      assertSame(document0, element0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Document document0 = new Document("exarea");
      Elements elements0 = document0.getElementsMatchingText("br");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Document document0 = new Document("col");
      Element element0 = document0.prependElement("col");
      element0.text("col");
      String string0 = document0.html();
      assertEquals("<col>\n col\n</col>", string0);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Document document0 = new Document("Dd}oc~9");
      document0.prependElement("br");
      document0.prependText("-#djb>)Zm': ");
      Elements elements0 = document0.getElementsContainingText("fRE N-cZv");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Document document0 = new Document("encoding");
      // Undeclared exception!
      try { 
        document0.child(4171);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 4171, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Document document0 = new Document("textarea");
      document0.tagName("textarea");
      document0.prependElement("textarea");
      String string0 = document0.html();
      assertEquals("<textarea></textarea>", string0);
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Document document0 = new Document("col");
      Map<String, String> map0 = document0.dataset();
      assertTrue(map0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Element element0 = new Element("br");
      element0.prependElement("br");
      element0.prependText("LHy");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      MockFileWriter mockFileWriter0 = new MockFileWriter("br");
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.outline(true);
      element0.outerHtmlTail(mockFileWriter0, 807, document_OutputSettings1);
      assertEquals(Document.OutputSettings.Syntax.html, document_OutputSettings1.syntax());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Document document0 = new Document("**J:%H]:a8f.mJ8");
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
      Document document0 = new Document("encoting");
      Element element0 = document0.attr("encoting", true);
      assertEquals(0, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Document document0 = new Document("Z_g)");
      Elements elements0 = document0.getElementsByAttributeValue("Z_g)", "Z_g)");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Document document0 = new Document("textarea");
      Elements elements0 = document0.getElementsByAttributeValueStarting("textarea", "textarea");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Document document0 = new Document("textarea");
      // Undeclared exception!
      try { 
        document0.after("textarea");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Document document0 = new Document("%Yh?TBE'K9:");
      Elements elements0 = document0.getElementsByIndexLessThan(69);
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Document document0 = new Document("7ool");
      Elements elements0 = document0.getElementsByAttributeStarting("7ool");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Document document0 = new Document("encoting");
      Elements elements0 = document0.getElementsByIndexEquals(46);
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Document document0 = new Document("encoVting");
      // Undeclared exception!
      try { 
        document0.wrap("encoVting");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Node", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Document document0 = new Document("q^IZ[v}}nN}<.");
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
  public void test18()  throws Throwable  {
      Document document0 = new Document("col");
      Document document1 = (Document)document0.val("col");
      assertFalse(document1.updateMetaCharsetElement());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Document document0 = new Document("col");
      Elements elements0 = document0.getElementsByAttributeValueMatching("col", "col");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Document document0 = new Document("X75g22jD");
      boolean boolean0 = document0.is("X75g22jD");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Document document0 = new Document("Should not be reachable");
      Elements elements0 = document0.getElementsByAttributeValueEnding("Should not be reachable", "Should not be reachable");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Document document0 = new Document("m.G#C#m|*_)");
      Elements elements0 = document0.getElementsByClass("m.G#C#m|*_)");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Document document0 = new Document("q");
      Elements elements0 = document0.getElementsByAttributeValueContaining("q", "q");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Document document0 = new Document("75g2aj");
      // Undeclared exception!
      try { 
        document0.after((Node) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Document document0 = new Document("figcaption");
      Elements elements0 = document0.getElementsByAttributeValueNot("figcaption", "figcaption");
      assertFalse(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Document document0 = new Document(" ");
      String string0 = document0.toString();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Document document0 = new Document("V");
      Elements elements0 = document0.getElementsByAttribute("V");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Document document0 = new Document("#root");
      // Undeclared exception!
      try { 
        document0.before("#root");
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
      Document document0 = new Document("l:IrqesT0Hl");
      Element element0 = document0.removeClass("l:IrqesT0Hl");
      assertEquals(0, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Document document0 = new Document("75g2aj");
      Elements elements0 = document0.getAllElements();
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Document document0 = new Document("figcaption");
      // Undeclared exception!
      try { 
        document0.title("figcaption");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Document", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Document document0 = new Document("}JU~Tl<\"YgrZj{Z/");
      Elements elements0 = document0.getElementsByIndexGreaterThan(1927);
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Tag tag0 = Tag.valueOf("<");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "<", attributes0);
      Element element0 = formElement0.prependElement("to");
      // Undeclared exception!
      try { 
        element0.html("to");
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
      Document document0 = new Document("col");
      Element element0 = document0.prependElement("col");
      Elements elements0 = element0.parents();
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Document document0 = new Document("em");
      Element element0 = document0.prependElement("em");
      document0.prependText("em");
      Element element1 = element0.lastElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Document document0 = new Document("textarea");
      document0.prependElement("br");
      List<TextNode> list0 = document0.textNodes();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Document document0 = new Document("!");
      Element element0 = document0.createElement("s+");
      element0.text("!");
      List<TextNode> list0 = element0.textNodes();
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Document document0 = new Document("col");
      document0.prependElement("col");
      List<DataNode> list0 = document0.dataNodes();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Document document0 = new Document("aT~(p|");
      DataNode dataNode0 = new DataNode("aT~(p|", "6#HQ5}FC>");
      document0.prependChild(dataNode0);
      List<DataNode> list0 = document0.dataNodes();
      assertTrue(list0.contains(dataNode0));
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Document document0 = new Document(",o");
      LinkedHashSet<TextNode> linkedHashSet0 = new LinkedHashSet<TextNode>();
      // Undeclared exception!
      try { 
        document0.insertChildren(7458, linkedHashSet0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Insert position out of bounds.
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Document document0 = new Document("col");
      LinkedHashSet<FormElement> linkedHashSet0 = new LinkedHashSet<FormElement>();
      // Undeclared exception!
      try { 
        document0.insertChildren((-2848), linkedHashSet0);
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
      Document document0 = new Document("nW");
      LinkedList<Comment> linkedList0 = new LinkedList<Comment>();
      Element element0 = document0.insertChildren(0, linkedList0);
      assertSame(element0, document0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Document document0 = new Document("75g2aj");
      Element element0 = document0.addClass("75g2aj");
      String string0 = element0.cssSelector();
      assertNotNull(string0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Document document0 = new Document("class");
      Node[] nodeArray0 = new Node[2];
      nodeArray0[0] = (Node) document0;
      nodeArray0[1] = (Node) document0;
      document0.addChildren(nodeArray0);
      String string0 = document0.cssSelector();
      assertEquals("#root", string0);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Tag tag0 = Tag.valueOf("3sPs[");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "3sPs[", attributes0);
      Element element0 = formElement0.prependElement("org.jsoup.nodes.Element");
      String string0 = element0.cssSelector();
      assertEquals("3sPs[ > org.jsoup.nodes.Element", string0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Document document0 = new Document("r");
      Elements elements0 = document0.siblingElements();
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Document document0 = new Document("figcaption");
      Document document1 = document0.clone();
      Node[] nodeArray0 = new Node[2];
      nodeArray0[0] = (Node) document0;
      nodeArray0[1] = (Node) document1;
      document1.addChildren(nodeArray0);
      Elements elements0 = document1.siblingElements();
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Document document0 = new Document("col");
      Element element0 = document0.prependElement("col");
      Element element1 = element0.nextElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Document document0 = new Document("col");
      Element element0 = document0.nextElementSibling();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Document document0 = new Document("figcpto");
      Document document1 = document0.clone();
      Node[] nodeArray0 = new Node[2];
      nodeArray0[0] = (Node) document0;
      nodeArray0[1] = (Node) document1;
      document1.addChildren(nodeArray0);
      Element element0 = document0.nextElementSibling();
      assertNotNull(element0);
      assertEquals(1, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Document document0 = new Document("class");
      Node[] nodeArray0 = new Node[2];
      nodeArray0[0] = (Node) document0;
      nodeArray0[1] = (Node) document0;
      document0.addChildren(nodeArray0);
      Element element0 = document0.previousElementSibling();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Document document0 = new Document("figcaption");
      Element element0 = document0.previousElementSibling();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Document document0 = new Document("class");
      document0.prependElement("class");
      Node[] nodeArray0 = new Node[2];
      nodeArray0[0] = (Node) document0;
      nodeArray0[1] = (Node) document0;
      document0.addChildren(nodeArray0);
      Element element0 = document0.previousElementSibling();
      assertEquals("class", element0.baseUri());
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Document document0 = new Document("col");
      Element element0 = document0.prependElement("col");
      Element element1 = element0.firstElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Document document0 = new Document("figcaption");
      Document document1 = document0.clone();
      Node[] nodeArray0 = new Node[2];
      nodeArray0[0] = (Node) document0;
      nodeArray0[1] = (Node) document1;
      document1.addChildren(nodeArray0);
      Element element0 = document0.firstElementSibling();
      assertNotNull(element0);
      assertEquals(0, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Document document0 = new Document("col");
      Element element0 = document0.prependElement("col");
      Integer integer0 = element0.elementSiblingIndex();
      assertEquals(0, (int)integer0);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Document document0 = new Document("figcaption");
      Document document1 = document0.clone();
      Node[] nodeArray0 = new Node[2];
      nodeArray0[0] = (Node) document0;
      nodeArray0[1] = (Node) document1;
      document1.addChildren(nodeArray0);
      Element element0 = document1.lastElementSibling();
      assertEquals(1, element0.siblingIndex());
      assertNotNull(element0);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Document document0 = new Document("75g2aj");
      document0.setParentNode(document0);
      // Undeclared exception!
      try { 
        document0.previousElementSibling();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Document document0 = new Document("7ool");
      Element element0 = document0.getElementById("7ool");
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Document document0 = new Document("col");
      document0.prependElement("col");
      Element element0 = document0.prependText("col");
      Elements elements0 = element0.getElementsContainingText("col");
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      Document document0 = new Document("encoVting");
      Node[] nodeArray0 = new Node[9];
      nodeArray0[0] = (Node) document0;
      nodeArray0[1] = (Node) document0;
      nodeArray0[2] = (Node) document0;
      DataNode dataNode0 = new DataNode("encoVting", "G4LBZ");
      nodeArray0[3] = (Node) dataNode0;
      nodeArray0[4] = (Node) document0;
      nodeArray0[5] = (Node) document0;
      nodeArray0[6] = (Node) document0;
      nodeArray0[7] = (Node) document0;
      nodeArray0[8] = (Node) document0;
      document0.addChildren(nodeArray0);
      Element element0 = document0.prependText("encoVting");
      element0.getElementsContainingText("CO|t1V-Q:");
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      Document document0 = new Document("75g2aj");
      document0.prepend("75g2aj");
      Elements elements0 = document0.getElementsContainingOwnText("75g2aj");
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      Element element0 = new Element("75g2aj");
      Node[] nodeArray0 = new Node[5];
      nodeArray0[0] = (Node) element0;
      Comment comment0 = new Comment("75g2aj", "75g2aj");
      nodeArray0[1] = (Node) comment0;
      nodeArray0[2] = (Node) element0;
      nodeArray0[3] = (Node) element0;
      nodeArray0[4] = (Node) element0;
      element0.addChildren(nodeArray0);
      // Undeclared exception!
      element0.getElementsContainingOwnText("");
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Document document0 = new Document("textarea");
      Element element0 = document0.prependElement("textarea");
      element0.prependText("textarea");
      Elements elements0 = element0.getElementsContainingText("textarea");
      assertFalse(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      Document document0 = new Document("textarea");
      document0.prependElement("br");
      Elements elements0 = document0.getElementsMatchingOwnText("textarea");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      boolean boolean0 = Element.preserveWhitespace((Node) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      Comment comment0 = new Comment("'", "'");
      boolean boolean0 = Element.preserveWhitespace(comment0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      Tag tag0 = Tag.valueOf("textarea");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "textarea", attributes0);
      Element element0 = formElement0.prependElement("br");
      boolean boolean0 = Element.preserveWhitespace(element0);
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      Document document0 = new Document("col");
      document0.prependElement("col");
      boolean boolean0 = document0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      Document document0 = new Document("75g2aj");
      Element element0 = document0.prependElement("75g2aj");
      element0.text("75g2aj");
      boolean boolean0 = document0.hasText();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      Document document0 = new Document("encoding");
      document0.prepend(" ");
      boolean boolean0 = document0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      Document document0 = new Document("l");
      DocumentType documentType0 = new DocumentType("l", "l", "l", "org.jsoup.parser.Token$EOF", "l");
      document0.prependChild(documentType0);
      boolean boolean0 = document0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test73()  throws Throwable  {
      Document document0 = new Document("encoVting");
      DataNode dataNode0 = new DataNode("y&Zn 9.2D`-~", "br");
      document0.appendChild(dataNode0);
      String string0 = document0.data();
      assertEquals("y&Zn 9.2D`-~", string0);
  }

  @Test(timeout = 4000)
  public void test74()  throws Throwable  {
      Document document0 = new Document("Pattern syntax error: ");
      Comment comment0 = new Comment("1W\"e*%3'@F{S", "i}o");
      Node[] nodeArray0 = new Node[7];
      nodeArray0[0] = (Node) comment0;
      nodeArray0[1] = (Node) comment0;
      nodeArray0[2] = (Node) document0;
      nodeArray0[3] = (Node) document0;
      nodeArray0[4] = (Node) document0;
      nodeArray0[5] = (Node) document0;
      nodeArray0[6] = (Node) document0;
      document0.addChildren(nodeArray0);
      // Undeclared exception!
      try { 
        document0.data();
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
      }
  }

  @Test(timeout = 4000)
  public void test75()  throws Throwable  {
      Document document0 = new Document("75g2aj");
      Element element0 = document0.prepend("75g2aj");
      String string0 = element0.data();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test76()  throws Throwable  {
      Document document0 = new Document("encoting");
      document0.toggleClass("    ");
      boolean boolean0 = document0.hasClass("java.lang.string@0000000007 children collection rp be inserted must not be null.");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test77()  throws Throwable  {
      Document document0 = new Document("75g2aj");
      document0.addClass("org.jsoup.helper.stringutil");
      document0.toggleClass("d>qC");
      boolean boolean0 = document0.hasClass("character outside of valid range");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test78()  throws Throwable  {
      Document document0 = new Document("em");
      document0.addClass("Pattern syntax error: ");
      document0.toggleClass("Pattern syntax error: ");
      boolean boolean0 = document0.hasClass("GR0f");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test79()  throws Throwable  {
      Document document0 = new Document("P7flO73R)3n+t5^4");
      document0.addClass("P7flO73R)3n+t5^4");
      document0.toggleClass(" />");
      boolean boolean0 = document0.hasClass("P7flO73R)3n+t5^4");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test80()  throws Throwable  {
      Document document0 = new Document("encoting");
      document0.addClass("Children collection to be inserted must not be null.");
      document0.toggleClass("Children collection to be inserted must not be null.");
      boolean boolean0 = document0.hasClass("rp");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test81()  throws Throwable  {
      Document document0 = new Document("z:[(IBA>");
      document0.toggleClass("z:[(IBA>");
      document0.addClass("textarea");
      document0.toggleClass("z:[(IBA>");
      boolean boolean0 = document0.hasClass("textarea");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test82()  throws Throwable  {
      Document document0 = new Document("encoting");
      document0.addClass("encoting");
      document0.toggleClass("encoting");
      boolean boolean0 = document0.hasClass("encoting");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test83()  throws Throwable  {
      Document document0 = new Document("75g2aj");
      document0.addClass("75g2aj");
      Element element0 = document0.toggleClass("75g2aj");
      Element element1 = document0.toggleClass("75g2aj");
      assertSame(element1, element0);
  }

  @Test(timeout = 4000)
  public void test84()  throws Throwable  {
      Document document0 = new Document("Pattern syntax error: ");
      String string0 = document0.val();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test85()  throws Throwable  {
      Document document0 = new Document("textarea");
      Element element0 = document0.prependElement("textarea");
      String string0 = element0.val();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test86()  throws Throwable  {
      Document document0 = new Document("textarea");
      Element element0 = document0.prependElement("textarea");
      element0.val("^S]PVIkB))xC");
      assertEquals(1, element0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test87()  throws Throwable  {
      Document document0 = new Document("col");
      MockFileWriter mockFileWriter0 = new MockFileWriter("col", false);
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.prettyPrint(false);
      document0.outerHtmlHead(mockFileWriter0, 34, document_OutputSettings1);
      assertEquals(0, document0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test88()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      MockFileWriter mockFileWriter0 = new MockFileWriter("75g2aj");
      Tag tag0 = Tag.valueOf("em");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "75g2aj", attributes0);
      formElement0.outerHtmlHead(mockFileWriter0, 740, document_OutputSettings0);
      assertEquals("75g2aj", formElement0.baseUri());
  }

  @Test(timeout = 4000)
  public void test89()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      MockFileWriter mockFileWriter0 = new MockFileWriter("75g2aj");
      Tag tag0 = Tag.valueOf("em");
      document_OutputSettings0.outline(true);
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "75g2aj", attributes0);
      formElement0.outerHtmlHead(mockFileWriter0, 740, document_OutputSettings0);
      assertEquals("75g2aj", formElement0.baseUri());
  }

  @Test(timeout = 4000)
  public void test90()  throws Throwable  {
      Document document0 = new Document("encoding");
      document0.prependElement("encoding");
      Element element0 = document0.prependText("encoding");
      String string0 = element0.html();
      assertEquals("encoding\n<encoding></encoding>", string0);
  }

  @Test(timeout = 4000)
  public void test91()  throws Throwable  {
      Document document0 = new Document("col");
      document0.prependElement("col");
      String string0 = document0.html();
      assertEquals("<col>", string0);
  }

  @Test(timeout = 4000)
  public void test92()  throws Throwable  {
      Document document0 = new Document("exarea");
      Element element0 = document0.prependElement("br");
      MockFileWriter mockFileWriter0 = new MockFileWriter("br", false);
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings.Syntax document_OutputSettings_Syntax0 = Document.OutputSettings.Syntax.xml;
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.syntax(document_OutputSettings_Syntax0);
      element0.outerHtmlHead(mockFileWriter0, 829, document_OutputSettings1);
      assertEquals("exarea", element0.baseUri());
  }

  @Test(timeout = 4000)
  public void test93()  throws Throwable  {
      Document document0 = new Document("em");
      MockFileWriter mockFileWriter0 = new MockFileWriter("9r&q6LsZ");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.prettyPrint(false);
      document0.outerHtmlTail(mockFileWriter0, (-1749), document_OutputSettings1);
      assertEquals("#document", document0.nodeName());
  }

  @Test(timeout = 4000)
  public void test94()  throws Throwable  {
      Document document0 = new Document("textarea");
      Element element0 = document0.prependElement("textarea");
      element0.text("textarea");
      String string0 = document0.html();
      assertEquals("<textarea>textarea</textarea>", string0);
  }

  @Test(timeout = 4000)
  public void test95()  throws Throwable  {
      Document document0 = new Document("Dd}oc~9");
      Element element0 = document0.prependElement("br");
      element0.prependText("Dd}oc~9");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      MockFileWriter mockFileWriter0 = new MockFileWriter("br");
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.outline(true);
      element0.outerHtmlTail(mockFileWriter0, 836, document_OutputSettings1);
      assertEquals("br", element0.nodeName());
  }

  @Test(timeout = 4000)
  public void test96()  throws Throwable  {
      Tag tag0 = Tag.valueOf("br");
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "s+", attributes0);
      formElement0.prependElement("qJ/LKgv3lNul");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      MockFileWriter mockFileWriter0 = new MockFileWriter("KPww6tN{q^WaeFP3t");
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.outline(true);
      formElement0.outerHtmlTail(mockFileWriter0, 58, document_OutputSettings1);
      assertTrue(document_OutputSettings1.outline());
  }

  @Test(timeout = 4000)
  public void test97()  throws Throwable  {
      Document document0 = new Document("aqY]KCAA`oV}");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.prettyPrint(false);
      Document document1 = document0.outputSettings(document_OutputSettings1);
      String string0 = document1.html();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test98()  throws Throwable  {
      Document document0 = new Document("f");
      MockPrintStream mockPrintStream0 = new MockPrintStream("f");
      MockPrintStream mockPrintStream1 = document0.html(mockPrintStream0);
      assertSame(mockPrintStream1, mockPrintStream0);
  }

  @Test(timeout = 4000)
  public void test99()  throws Throwable  {
      Document document0 = new Document("textarea");
      document0.prependElement("br");
      // Undeclared exception!
      try { 
        document0.html((MockPrintStream) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Node", e);
      }
  }
}
