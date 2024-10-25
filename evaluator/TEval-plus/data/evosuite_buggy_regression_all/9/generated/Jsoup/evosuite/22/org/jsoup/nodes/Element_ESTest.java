/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:04:38 GMT 2023
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
      Tag tag0 = Tag.valueOf("br");
      Element element0 = new Element(tag0, "br");
      element0.prependText("br");
      String string0 = element0.text();
      assertEquals("br", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Document document0 = new Document("#ldoctype");
      document0.appendText("#ldoctype");
      String string0 = document0.data();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Document document0 = new Document("gdItLJ");
      Elements elements0 = document0.getElementsMatchingText("gdItLJ");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Tag tag0 = Tag.valueOf("br");
      Element element0 = new Element(tag0, "br");
      Element element1 = element0.prependElement("br");
      element0.text();
      assertNotSame(element0, element1);
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Tag tag0 = Tag.valueOf("br");
      Element element0 = new Element(tag0, "br");
      element0.prependElement("br");
      // Undeclared exception!
      try { 
        element0.toString();
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Document document0 = new Document("#doctype");
      Document document1 = (Document)document0.tagName("#doctype");
      assertEquals(Document.QuirksMode.noQuirks, document1.quirksMode());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Document document0 = new Document("textarea");
      Element element0 = document0.prependElement("textarea");
      Element element1 = element0.val("textarea");
      assertSame(element0, element1);
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Document document0 = new Document("br");
      Map<String, String> map0 = document0.dataset();
      assertTrue(map0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Document document0 = new Document("eY_'f/--X+b+?6(ze0I");
      Element element0 = document0.addClass("zB*B4o7OKr[q[]\"(p");
      Element element1 = element0.toggleClass("Nnk$DK*pyg");
      assertSame(element1, element0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Document document0 = new Document("#'do:;typm");
      // Undeclared exception!
      try { 
        document0.html("#'do:;typm");
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
      Document document0 = new Document("RKUy");
      Elements elements0 = document0.getElementsByAttributeValue("org.jsoup.nodes.Element", " w(2#u\"|'+");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Document document0 = new Document("#doctype");
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
      Document document0 = new Document("h`>AqlT8");
      Elements elements0 = document0.getElementsByAttributeValueStarting("h`>AqlT8", "h`>AqlT8");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Document document0 = new Document("#ldoctype");
      Elements elements0 = document0.select("#ldoctype");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Document document0 = new Document("#'docty pe");
      Element element0 = document0.prepend("#'docty pe");
      assertEquals("#'docty pe", element0.baseUri());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Document document0 = new Document("#'do:;typm");
      Elements elements0 = document0.getElementsMatchingOwnText("#'do:;typm");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Document document0 = new Document("#ldoctHype");
      // Undeclared exception!
      try { 
        document0.after("#ldoctHype");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Document document0 = new Document("sect");
      Elements elements0 = document0.getElementsByIndexLessThan((-1434424480));
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Document document0 = new Document(" !pGa(tD<iC<");
      Elements elements0 = document0.getElementsByAttributeStarting(" !pGa(tD<iC<");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Document document0 = new Document("#'docty pe");
      Elements elements0 = document0.getElementsByIndexEquals(621);
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Document document0 = new Document("#doctype");
      // Undeclared exception!
      try { 
        document0.wrap("#doctype");
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
      Document document0 = new Document("Rarrtl");
      Elements elements0 = document0.getElementsByAttributeValueMatching("Rarrtl", "Rarrtl");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Document document0 = new Document("swnwar");
      Elements elements0 = document0.getElementsByAttributeValueEnding("swnwar", "swnwar");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Document document0 = new Document("#doctype");
      Elements elements0 = document0.getElementsByClass("#doctype");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Document document0 = new Document("xrea");
      Elements elements0 = document0.getElementsByAttributeValueContaining("xrea", "xrea");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Document document0 = new Document("br");
      Document document1 = document0.clone();
      assertNotSame(document1, document0);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Document document0 = new Document("#doctype");
      Elements elements0 = document0.getElementsByAttributeValueNot("#doctype", "#doctype");
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Document document0 = new Document("BRfECV'z<'LR6[<1YNA");
      Elements elements0 = document0.getElementsByAttribute("BRfECV'z<'LR6[<1YNA");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Document document0 = new Document("b*");
      // Undeclared exception!
      try { 
        document0.before("b*");
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
      Document document0 = new Document("#doctype");
      Element element0 = document0.removeClass("#doctype");
      assertEquals(0, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Document document0 = new Document("d<7U%6)k]=5K9>>;D");
      Elements elements0 = document0.getAllElements();
      assertFalse(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Document document0 = new Document("6uYT`ux_4vZY");
      // Undeclared exception!
      try { 
        document0.title("6uYT`ux_4vZY");
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
      Document document0 = new Document("#doctype");
      Elements elements0 = document0.getElementsByIndexGreaterThan((-1));
      assertFalse(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Tag tag0 = Tag.valueOf("br");
      Element element0 = new Element(tag0, "br");
      Element element1 = element0.prependElement("br");
      Elements elements0 = element1.parents();
      assertFalse(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Document document0 = new Document("org.jsoup.nodes.document$outputsettings");
      Element element0 = document0.prependElement("org.jsoup.nodes.document$outputsettings");
      Elements elements0 = element0.parents();
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Tag tag0 = Tag.valueOf("?7I}w[");
      Element element0 = new Element(tag0, "s+");
      Document document0 = new Document("eY_'f/--X+b+?6(ze0I");
      element0.appendChild(document0);
      Element element1 = document0.lastElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Document document0 = new Document("#doctype");
      Element element0 = document0.append("#doctype");
      // Undeclared exception!
      try { 
        element0.child(31);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 31, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Document document0 = new Document("#'doctype");
      Element element0 = document0.prependElement("U:!`nD:ib4U+-");
      assertEquals("u:!`nd:ib4u+-", element0.nodeName());
      
      List<TextNode> list0 = document0.textNodes();
      assertEquals(0, list0.size());
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Document document0 = new Document("#doctype");
      document0.append("#doctype");
      List<TextNode> list0 = document0.textNodes();
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Document document0 = new Document("#doctype");
      Element element0 = document0.append("#doctype");
      List<DataNode> list0 = element0.dataNodes();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Document document0 = new Document("#octtype");
      Element element0 = document0.prependElement("#octtype");
      Element element1 = element0.nextElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Document document0 = new Document("swnwar");
      Element element0 = document0.prependElement("swnwar");
      element0.after((Node) document0);
      Element element1 = element0.nextElementSibling();
      assertFalse(element1.isBlock());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Document document0 = new Document("textarea");
      Element element0 = document0.prependElement("textarea");
      Element element1 = element0.previousElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Document document0 = new Document("br");
      Element element0 = document0.createElement("Pattern syntax error: ");
      document0.prependChild(element0);
      document0.prependChild(document0);
      Element element1 = element0.previousElementSibling();
      assertSame(element1, document0);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Document document0 = new Document("#doctype");
      Element element0 = document0.prependElement("#doctype");
      Element element1 = element0.firstElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Document document0 = new Document("#doctype");
      Element element0 = document0.prependElement("#doctype");
      element0.before((Node) document0);
      Document document1 = (Document)element0.firstElementSibling();
      assertEquals(Document.QuirksMode.noQuirks, document1.quirksMode());
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Document document0 = new Document("f(");
      Element element0 = document0.prependElement("f(");
      Integer integer0 = element0.elementSiblingIndex();
      assertEquals(0, (int)integer0);
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Tag tag0 = Tag.valueOf("?7I}w[");
      Element element0 = new Element(tag0, "s+");
      Document document0 = new Document("eY_'f/--X+b+?6(ze0I");
      Element element1 = element0.appendChild(document0);
      Element element2 = element1.appendElement("s+");
      Element element3 = element2.lastElementSibling();
      assertNotSame(element1, element3);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Document document0 = new Document("#'do;typm");
      document0.parentNode = (Node) document0;
      // Undeclared exception!
      try { 
        document0.nextElementSibling();
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Document document0 = new Document("br");
      Element element0 = document0.getElementById("br");
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Document document0 = new Document("br");
      Element element0 = document0.append("<!u");
      String string0 = element0.text();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Document document0 = new Document("61zVl{CzO]vdm");
      document0.prependElement("61zVl{CzO]vdm");
      Elements elements0 = document0.getElementsContainingText("61zVl{CzO]vdm");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Document document0 = new Document("f(");
      document0.prependElement("f(");
      Elements elements0 = document0.getElementsContainingOwnText("f(");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Document document0 = new Document("#doctype");
      document0.append("#doctype");
      Elements elements0 = document0.getElementsContainingOwnText("#doctype");
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Document document0 = new Document("RKUy");
      Element element0 = document0.prependElement("RKUy");
      boolean boolean0 = element0.preserveWhitespace();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Document document0 = new Document("br");
      DocumentType documentType0 = new DocumentType("br", "br", "gydqraa:[r|p^", "dctype");
      Element element0 = document0.prependChild(documentType0);
      boolean boolean0 = element0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Document document0 = new Document("br");
      document0.appendText("");
      boolean boolean0 = document0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Document document0 = new Document("#doctype");
      document0.prependElement("#doctype");
      boolean boolean0 = document0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Document document0 = new Document("#doctype");
      Element element0 = document0.createElement(",O~N371U`F?=4vz$y");
      Element element1 = element0.val(",O~N371U`F?=4vz$y");
      document0.append("rawtextendtagopen");
      Element element2 = element1.prependChild(document0);
      boolean boolean0 = element2.hasText();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Document document0 = new Document("textarea");
      document0.prependElement("textarea");
      String string0 = document0.data();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Tag tag0 = Tag.valueOf("br");
      Element element0 = new Element(tag0, "br");
      Element element1 = element0.addClass("br");
      boolean boolean0 = element1.hasClass("br");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      Document document0 = new Document("");
      Element element0 = document0.toggleClass("");
      assertEquals("", element0.baseUri());
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      Document document0 = new Document("br");
      String string0 = document0.val();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      Document document0 = new Document("+ucucH-x5z");
      Element element0 = document0.createElement("textarea");
      String string0 = element0.val();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Document document0 = new Document("RKUy");
      StringBuilder stringBuilder0 = new StringBuilder("RKUy");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.prettyPrint(false);
      document0.outerHtmlHead(stringBuilder0, (-1), document_OutputSettings1);
      assertEquals("RKUy<#root>", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      Tag tag0 = Tag.valueOf("G5UR}g||S:");
      Attributes attributes0 = new Attributes();
      Element element0 = new Element(tag0, "G5UR}g||S:", attributes0);
      StringBuilder stringBuilder0 = new StringBuilder("InSelectInTable");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      document_OutputSettings0.prettyPrint(false);
      element0.outerHtmlTail(stringBuilder0, (-2342), document_OutputSettings0);
      assertEquals("InSelectInTable</g5ur}g||s:>", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      Tag tag0 = Tag.valueOf("#octtype");
      Element element0 = new Element(tag0, "#octtype");
      element0.prependElement("#octtype");
      // Undeclared exception!
      try { 
        element0.toString();
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      Document document0 = new Document("#doctype");
      document0.prependElement("#doctype");
      String string0 = document0.toString();
      assertEquals("<#doctype></#doctype>", string0);
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      Document document0 = new Document("gdItLJ");
      document0.hashCode();
  }
}
