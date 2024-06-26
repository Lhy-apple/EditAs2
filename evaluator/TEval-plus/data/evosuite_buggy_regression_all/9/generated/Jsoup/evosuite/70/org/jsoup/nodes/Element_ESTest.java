/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:14:13 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.io.CharArrayWriter;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.regex.Pattern;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Attributes;
import org.jsoup.nodes.Comment;
import org.jsoup.nodes.DataNode;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.DocumentType;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.FormElement;
import org.jsoup.nodes.Node;
import org.jsoup.parser.ParseSettings;
import org.jsoup.parser.Tag;
import org.jsoup.select.Elements;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Element_ESTest extends Element_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Document document0 = Document.createShell("SelfClosingStartTag");
      // Undeclared exception!
      try { 
        document0.wrap("SelfClosingStartTag");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Node", e);
      }
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Document document0 = Document.createShell("java.lang.string@0000000006 ie");
      document0.appendText("java.lang.string@0000000006 ie");
      document0.getElementsByIndexEquals((-1983));
      assertEquals(2, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Element element0 = new Element("br");
      element0.prependChild(element0);
      element0.getElementsByIndexEquals(1);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Document document0 = Document.createShell("br");
      Element element0 = document0.text("br");
      element0.appendTo(document0);
      document0.text();
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Document document0 = Document.createShell("FZID>q,3");
      // Undeclared exception!
      try { 
        document0.child(2185);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 2185, Size: 1
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Document document0 = new Document("O'6BgARt?,9V=7a<");
      document0.prependText("O'6BgARt?,9V=7a<");
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Element element0 = new Element("textarea");
      Element element1 = element0.tagName("textarea");
      assertEquals(0, element1.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Document document0 = Document.createShell("java.lang.string@0000000011 :$!hfavu0o n;:%`");
      document0.dataset();
      assertFalse(document0.hasParent());
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Element element0 = new Element("meta");
      Element element1 = element0.addClass("meta");
      assertEquals(0, element1.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Element element0 = new Element("s");
      element0.html("s");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.outline(true);
      // Undeclared exception!
      try { 
        element0.outerHtmlTail((Appendable) null, 2125, document_OutputSettings1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Element", e);
      }
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Document document0 = Document.createShell("meta");
      Element element0 = document0.attr("meta", false);
      assertEquals("meta", element0.baseUri());
      assertEquals(1, element0.childNodeSize());
      assertFalse(element0.hasParent());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Element element0 = new Element("br");
      Elements elements0 = element0.getElementsByAttributeValue("Xz", "br");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Element element0 = new Element("{XA@,-YO");
      Elements elements0 = element0.getElementsByAttributeValueStarting("{XA@,-YO", "{XA@,-YO");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Document document0 = Document.createShell("#a");
      assertFalse(document0.hasParent());
      
      Element element0 = document0.head();
      String string0 = element0.cssSelector();
      assertEquals("html > head", string0);
      assertEquals(1, document0.childNodeSize());
      assertEquals("#a", element0.baseUri());
      assertEquals(0, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Document document0 = Document.createShell("org.jsoup.nodes.Element$1");
      assertFalse(document0.hasParent());
      
      Elements elements0 = document0.getElementsMatchingOwnText("org.jsoup.nodes.Element$1");
      assertEquals(1, document0.childNodeSize());
      assertTrue(elements0.isEmpty());
      assertEquals("org.jsoup.nodes.Element$1", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Element element0 = new Element("br");
      // Undeclared exception!
      try { 
        element0.after("br");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Document document0 = new Document("xt");
      document0.getElementsByIndexLessThan(0);
      assertEquals("xt", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Document document0 = new Document("pE");
      document0.normalise();
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Document document0 = new Document("textarea");
      document0.getElementsByAttributeStarting("textarea");
      assertEquals("textarea", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Element element0 = new Element("g>W!PEun)> n;:%`");
      Element element1 = element0.val("g>W!PEun)> n;:%`");
      assertEquals(0, element1.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Document document0 = Document.createShell("button");
      document0.getElementsByAttributeValueMatching("button", "button");
      assertEquals("button", document0.baseUri());
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Document document0 = Document.createShell("meta");
      document0.is("meta");
      assertEquals(1, document0.childNodeSize());
      assertEquals("meta", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Element element0 = new Element("s");
      Elements elements0 = element0.getElementsByAttributeValueEnding("s", "s");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Document document0 = Document.createShell("6U.");
      Elements elements0 = document0.getElementsByClass("6U.");
      assertEquals("6U.", document0.baseUri());
      assertEquals(0, elements0.size());
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Element element0 = new Element("textarea");
      Elements elements0 = element0.getElementsContainingText("textarea");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Document document0 = Document.createShell("6U.");
      document0.getElementsByAttributeValueContaining("6U.", "6U.");
      assertEquals("6U.", document0.baseUri());
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Document document0 = Document.createShell(":nth-child(%)");
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
      Element element0 = new Element("meta");
      element0.doSetBaseUri("meta");
      assertEquals("meta", element0.baseUri());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Element element0 = new Element("b>TzV7reyX");
      Element element1 = element0.clone();
      assertNotSame(element1, element0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Document document0 = Document.createShell("6U.");
      Element element0 = document0.shallowClone();
      assertEquals("6U.", element0.baseUri());
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Document document0 = Document.createShell("6U.");
      Elements elements0 = document0.getElementsByAttributeValueNot("6U.", "6U.");
      assertEquals(4, elements0.size());
      assertEquals("6U.", document0.baseUri());
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Document document0 = new Document("br");
      document0.getElementsByAttribute("br");
      assertEquals("br", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Element element0 = new Element("br");
      Element element1 = element0.removeClass("br");
      assertEquals("", element1.baseUri());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Document document0 = Document.createShell("meta");
      Elements elements0 = document0.getAllElements();
      assertEquals(1, document0.childNodeSize());
      assertEquals(4, elements0.size());
      assertEquals("meta", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Document document0 = new Document("meta");
      Elements elements0 = document0.getElementsMatchingText("meta");
      assertTrue(elements0.isEmpty());
      assertEquals("meta", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Document document0 = Document.createShell("Insert position out of bounds.");
      // Undeclared exception!
      try { 
        document0.selectFirst("g>W!PEun)> n;:%`");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Could not parse query 'W!PEun)': unexpected token at '!PEun)'
         //
         verifyException("org.jsoup.select.QueryParser", e);
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Document document0 = new Document("6*.");
      Elements elements0 = document0.getElementsByIndexGreaterThan(224);
      assertEquals("6*.", document0.baseUri());
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Document document0 = Document.createShell("#a");
      Element element0 = document0.head();
      assertEquals(0, element0.siblingIndex());
      
      element0.before("#a");
      String string0 = element0.cssSelector();
      assertEquals("html > head:nth-child(3)", string0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Document document0 = Document.createShell("meta");
      Element element0 = document0.head();
      element0.append("meta");
      assertEquals(1, element0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Document document0 = Document.createShell("g>W!PEun)> n;:%`");
      Element element0 = document0.prepend("llt+mG!&_[XaKj<e4(l");
      element0.textNodes();
      assertEquals(2, document0.childNodeSize());
      assertEquals("g>W!PEun)> n;:%`", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Document document0 = Document.createShell("g>W!PEun)> n;:%`");
      DataNode dataNode0 = new DataNode("<DMT92G$", "(*aw");
      document0.prependChild(dataNode0);
      document0.dataNodes();
      assertEquals(2, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Document document0 = Document.createShell(":nth-child(%)");
      LinkedHashSet<Element> linkedHashSet0 = new LinkedHashSet<Element>();
      // Undeclared exception!
      try { 
        document0.insertChildren(29, (Collection<? extends Node>) linkedHashSet0);
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
      Document document0 = Document.createShell("br");
      LinkedList<Comment> linkedList0 = new LinkedList<Comment>();
      // Undeclared exception!
      try { 
        document0.insertChildren((-863), (Collection<? extends Node>) linkedList0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Insert position out of bounds.
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Document document0 = Document.createShell("g>W!PEun)> n;:%`");
      LinkedHashSet<DocumentType> linkedHashSet0 = new LinkedHashSet<DocumentType>();
      Element element0 = document0.insertChildren(0, (Collection<? extends Node>) linkedHashSet0);
      assertEquals(1, element0.childNodeSize());
      assertEquals("g>W!PEun)> n;:%`", element0.baseUri());
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Document document0 = Document.createShell("y+G");
      Node[] nodeArray0 = new Node[0];
      // Undeclared exception!
      try { 
        document0.insertChildren(597, nodeArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Insert position out of bounds.
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Element element0 = new Element("[9W?#u40+Rh[(7n");
      Node[] nodeArray0 = new Node[2];
      // Undeclared exception!
      try { 
        element0.insertChildren((-4512), nodeArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Insert position out of bounds.
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Document document0 = new Document("g>WPEu`)> n;:%`");
      Node[] nodeArray0 = new Node[3];
      // Undeclared exception!
      try { 
        document0.insertChildren(0, nodeArray0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Array must not contain any null objects
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Document document0 = new Document("g>W!PEun)> n;:%`");
      Element element0 = document0.toggleClass("g>W!PEun)> n;:%`");
      element0.cssSelector();
      assertEquals("g>W!PEun)> n;:%`", element0.baseUri());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Document document0 = Document.createShell("g>W!PEun)> n;:%`");
      Element element0 = document0.body();
      Elements elements0 = element0.siblingElements();
      assertEquals("g>W!PEun)> n;:%`", element0.baseUri());
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Element element0 = new Element("meta");
      Elements elements0 = element0.siblingElements();
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Document document0 = Document.createShell("G6yF\"e:d.S");
      Element element0 = document0.body();
      Element element1 = element0.nextElementSibling();
      assertNull(element1);
      assertEquals(1, element0.siblingIndex());
      assertEquals("G6yF\"e:d.S", element0.baseUri());
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Document document0 = Document.createShell("G6yF\"e:d.S");
      document0.nextElementSibling();
      assertEquals("G6yF\"e:d.S", document0.baseUri());
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Document document0 = Document.createShell("g>W!PEun)> n;:%`");
      Element element0 = document0.body();
      Element element1 = element0.firstElementSibling();
      assertNotNull(element1);
      
      Element element2 = element1.nextElementSibling();
      assertNotNull(element2);
      assertEquals("g>W!PEun)> n;:%`", element2.baseUri());
      assertEquals("body", element2.nodeName());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Document document0 = new Document("{XA@,-6YPO");
      Element element0 = document0.appendElement("{XA@,-6YPO");
      Element element1 = element0.previousElementSibling();
      assertEquals("{XA@,-6YPO", element0.baseUri());
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Document document0 = Document.createShell("Mw~,(C@h)FQ[I-F<");
      document0.previousElementSibling();
      assertEquals("Mw~,(C@h)FQ[I-F<", document0.baseUri());
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Document document0 = Document.createShell("jJX WubT!/8ve|I");
      Element element0 = document0.body();
      Element element1 = element0.previousElementSibling();
      assertEquals(0, element1.siblingIndex());
      assertNotNull(element1);
      assertEquals("jJX WubT!/8ve|I", element1.baseUri());
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      ParseSettings parseSettings0 = ParseSettings.preserveCase;
      Tag tag0 = Tag.valueOf("org.jsoup.select.Evaluator$AttributeKeyPair", parseSettings0);
      Attributes attributes0 = new Attributes();
      FormElement formElement0 = new FormElement(tag0, "org.jsoup.select.Evaluator$AttributeKeyPair", attributes0);
      formElement0.appendTo(formElement0);
      Element element0 = formElement0.firstElementSibling();
      assertNull(element0);
      assertEquals("org.jsoup.select.Evaluator$AttributeKeyPair", formElement0.baseUri());
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Document document0 = Document.createShell("g>W!PEun)> n;:%`");
      document0.setParentNode(document0);
      Element element0 = document0.lastElementSibling();
      assertNull(element0);
      assertEquals("g>W!PEun)> n;:%`", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Document document0 = Document.createShell("g>W!PEun)> n;:%`");
      Element element0 = document0.body();
      Element element1 = element0.lastElementSibling();
      assertNotNull(element1);
      assertEquals("g>W!PEun)> n;:%`", element1.baseUri());
      assertEquals(1, element1.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Document document0 = new Document(":v)ns@:");
      document0.setParentNode(document0);
      Elements elements0 = document0.getElementsByIndexEquals((-2249));
      assertEquals(":v)ns@:", document0.baseUri());
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Document document0 = new Document("-b3?y|o}(p,|");
      Element element0 = document0.getElementById("-b3?y|o}(p,|");
      assertEquals("-b3?y|o}(p,|", document0.baseUri());
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      Document document0 = Document.createShell("hMrKp(8~'@U&);JP");
      Comment comment0 = new Comment((String) null);
      document0.appendChild(comment0);
      document0.text();
      assertEquals(2, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      Document document0 = Document.createShell("br");
      document0.appendElement("br");
      document0.text("V`t");
      document0.text();
      assertEquals(2, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      Document document0 = Document.createShell("g>w!peun)> n;:%`");
      Element element0 = document0.text("g>w!peun)> n;:%`");
      Elements elements0 = element0.getElementsContainingOwnText("g>w!peun)> n;:%`");
      assertEquals(1, elements0.size());
      assertEquals(1, element0.childNodeSize());
      assertEquals("g>w!peun)> n;:%`", element0.baseUri());
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Document document0 = Document.createShell(":nth-child(%)");
      Comment comment0 = new Comment(" > ", " > ");
      document0.appendChild(comment0);
      document0.getElementsContainingOwnText(":nth-child(%)");
      assertEquals(2, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      Document document0 = Document.createShell("R/ze;");
      document0.title("R/ze;");
      String string0 = document0.text();
      assertEquals("R/ze;", string0);
      assertEquals("R/ze;", document0.baseUri());
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      Document document0 = Document.createShell("br");
      ParseSettings parseSettings0 = ParseSettings.htmlDefault;
      Tag tag0 = Tag.valueOf("br", parseSettings0);
      Element element0 = new Element(tag0, "br");
      document0.prependChild(element0);
      document0.appendElement("br");
      document0.getElementsContainingOwnText("br");
      assertEquals(3, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      boolean boolean0 = Element.preserveWhitespace((Node) null);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      DataNode dataNode0 = new DataNode(")`;),");
      boolean boolean0 = Element.preserveWhitespace(dataNode0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      Element element0 = new Element("meta");
      element0.appendText("meta");
      String string0 = element0.toString();
      assertEquals("<meta>\n meta\n</meta>", string0);
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      Document document0 = Document.createShell("");
      Element element0 = document0.text("");
      boolean boolean0 = element0.hasText();
      assertFalse(boolean0);
      assertEquals(1, element0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      Document document0 = Document.createShell("marquee");
      Element element0 = document0.text("marquee");
      boolean boolean0 = element0.hasText();
      assertTrue(boolean0);
      assertEquals(1, element0.childNodeSize());
      assertEquals("marquee", element0.baseUri());
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      Document document0 = Document.createShell("inrt");
      DocumentType documentType0 = new DocumentType("inrt", "Header name must not be null", "WnoG nG");
      document0.prependChild(documentType0);
      boolean boolean0 = document0.hasText();
      assertEquals(2, document0.childNodeSize());
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test73()  throws Throwable  {
      Document document0 = Document.createShell("g>W!PEun)> n;:%`");
      DataNode dataNode0 = new DataNode("<DMT92G$", "(*aw");
      document0.prependChild(dataNode0);
      document0.data();
      assertEquals(2, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test74()  throws Throwable  {
      Document document0 = Document.createShell(":nth-child(%)");
      Comment comment0 = new Comment(" > ", " > ");
      Element element0 = document0.appendChild(comment0);
      String string0 = element0.data();
      assertEquals(1, comment0.siblingIndex());
      assertEquals(" > ", string0);
  }

  @Test(timeout = 4000)
  public void test75()  throws Throwable  {
      Document document0 = Document.createShell(":vP)1ns@:");
      document0.text(":vP)1ns@:");
      document0.data();
      assertEquals(":vP)1ns@:", document0.baseUri());
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test76()  throws Throwable  {
      Document document0 = Document.createShell("org.jsoup.nodes.Element$NodeList");
      Element element0 = document0.toggleClass("org.jsoup.nodes.Element$NodeList");
      boolean boolean0 = element0.hasClass("<html>\n <head>\n  <title>meta</title>\n </head>\n <body></body>\n</html>");
      assertEquals("org.jsoup.nodes.Element$NodeList", element0.baseUri());
      assertFalse(boolean0);
      assertEquals(1, element0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test77()  throws Throwable  {
      Document document0 = Document.createShell("java.lang.string@0000000018");
      document0.toggleClass("R/ze;");
      boolean boolean0 = document0.hasClass("java.lang.string@0000000018");
      assertFalse(boolean0);
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test78()  throws Throwable  {
      Document document0 = new Document("br");
      document0.toggleClass("br");
      Element element0 = document0.toggleClass("br");
      element0.hasClass("Ie");
      assertEquals("br", element0.baseUri());
  }

  @Test(timeout = 4000)
  public void test79()  throws Throwable  {
      Document document0 = new Document("7|]v?xA0 3g");
      Element element0 = document0.toggleClass("8Rq`-J}>9li4O.");
      document0.toggleClass(" ");
      boolean boolean0 = element0.hasClass("8Rq`-J}>9li4O.");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test80()  throws Throwable  {
      Document document0 = new Document("g>W!PEun)> n;:%`");
      document0.toggleClass("g>W!PEun)> n;:%`");
      document0.toggleClass("g>W!PEun)> n;:%`");
      boolean boolean0 = document0.hasClass("java.lang.string@0000000010");
      assertFalse(boolean0);
      assertEquals("g>W!PEun)> n;:%`", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test81()  throws Throwable  {
      Document document0 = new Document("g>W!PEun)> n;:%`");
      document0.toggleClass("g>W!PEun)> n;:%`");
      Pattern pattern0 = Pattern.compile("java.lang.string@0000000010", 124);
      document0.getElementsMatchingText(pattern0);
      document0.toggleClass("g>W!PEun)> n;:%`");
      boolean boolean0 = document0.hasClass("java.lang.string@0000000010");
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test82()  throws Throwable  {
      Document document0 = Document.createShell("{XA@,-YO");
      Element element0 = document0.toggleClass("{XA@,-YO");
      Element element1 = element0.toggleClass("{XA@,-YO");
      Element element2 = element1.toggleClass("{XA@,-YO");
      assertEquals("{XA@,-YO", element2.baseUri());
      assertEquals(1, element2.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test83()  throws Throwable  {
      Document document0 = Document.createShell("g>1!peun)> ynk;: `");
      document0.val();
      assertEquals("g>1!peun)> ynk;: `", document0.baseUri());
      assertEquals(1, document0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test84()  throws Throwable  {
      Element element0 = new Element("textarea");
      String string0 = element0.val();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test85()  throws Throwable  {
      Element element0 = new Element("textarea");
      element0.val("Yx; YDjOG");
      assertEquals(1, element0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test86()  throws Throwable  {
      Document document0 = Document.createShell("g>1!peun)> ynk;: `");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      document_OutputSettings0.prettyPrint(false);
      Document document1 = document0.outputSettings(document_OutputSettings0);
      String string0 = document1.toString();
      assertEquals("<html><head></head><body></body></html>", string0);
      assertEquals("g>1!peun)> ynk;: `", document1.baseUri());
  }

  @Test(timeout = 4000)
  public void test87()  throws Throwable  {
      Element element0 = new Element("br");
      Element element1 = element0.appendElement("br");
      String string0 = element0.toString();
      assertNotSame(element0, element1);
      assertEquals("<br><br></br>", string0);
  }

  @Test(timeout = 4000)
  public void test88()  throws Throwable  {
      Document document0 = Document.createShell("R/ze;");
      document0.title("R/ze;");
      String string0 = document0.toString();
      assertEquals("<html>\n <head>\n  <title>R/ze;</title>\n </head>\n <body></body>\n</html>", string0);
      assertEquals("R/ze;", document0.baseUri());
  }

  @Test(timeout = 4000)
  public void test89()  throws Throwable  {
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      document_OutputSettings0.outline(true);
      Element element0 = new Element("s");
      // Undeclared exception!
      try { 
        element0.outerHtmlHead((Appendable) null, 26, document_OutputSettings0);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Node", e);
      }
  }

  @Test(timeout = 4000)
  public void test90()  throws Throwable  {
      Element element0 = new Element("br");
      StringBuilder stringBuilder0 = new StringBuilder("br");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings.Syntax document_OutputSettings_Syntax0 = Document.OutputSettings.Syntax.xml;
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.syntax(document_OutputSettings_Syntax0);
      element0.outerHtmlHead(stringBuilder0, (-3627), document_OutputSettings1);
      assertEquals("br<br />", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test91()  throws Throwable  {
      Element element0 = new Element("s");
      element0.appendElement("s");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.outline(true);
      // Undeclared exception!
      try { 
        element0.outerHtmlTail((Appendable) null, 2125, document_OutputSettings1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Node", e);
      }
  }

  @Test(timeout = 4000)
  public void test92()  throws Throwable  {
      Element element0 = new Element("s");
      Element element1 = element0.appendElement("s");
      element1.before((Node) element0);
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.outline(true);
      // Undeclared exception!
      try { 
        element0.outerHtmlTail((Appendable) null, 2125, document_OutputSettings1);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Node", e);
      }
  }

  @Test(timeout = 4000)
  public void test93()  throws Throwable  {
      Document document0 = Document.createShell("br");
      CharArrayWriter charArrayWriter0 = new CharArrayWriter();
      document0.html(charArrayWriter0);
      assertEquals(45, charArrayWriter0.size());
      assertEquals("\n<html>\n <head></head>\n <body></body>\n</html>", charArrayWriter0.toString());
  }

  @Test(timeout = 4000)
  public void test94()  throws Throwable  {
      Document document0 = Document.createShell("br");
      Document document1 = document0.clone();
      assertEquals(1, document0.childNodeSize());
      assertEquals("br", document1.baseUri());
      assertNotSame(document1, document0);
  }
}
