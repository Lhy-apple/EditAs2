/*
 * This file was automatically generated by EvoSuite
 * Sat Jul 29 19:15:07 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.LinkedHashSet;
import java.util.NoSuchElementException;
import java.util.Set;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Attributes;
import org.jsoup.nodes.DataNode;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.XmlDeclaration;
import org.jsoup.parser.Tag;
import org.jsoup.select.Elements;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class Element_ESTest extends Element_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test00()  throws Throwable  {
      Document document0 = new Document("textarea");
      document0.prependElement("textarea");
      String string0 = document0.toString();
      assertEquals("<textarea></textarea>", string0);
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Document document0 = new Document("|,sapQ*J7f};");
      document0.appendText("|,sapQ*J7f};");
      Element element0 = document0.getElementById("|,sapQ*J7f};");
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test02()  throws Throwable  {
      Document document0 = new Document("4{p!7Oy+x.,mBx7");
      Element element0 = document0.val("4{p!7Oy+x.,mBx7");
      assertEquals("#root", element0.tagName());
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Document document0 = new Document("P0r~#JG");
      Elements elements0 = document0.getElementsByAttributeValueEnding("#comment", "4{p!7Oy+x.,mBx7");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Document document0 = new Document("%>c?>2S!yM");
      Elements elements0 = document0.getElementsByClass("%>c?>2S!yM");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Document document0 = new Document("4{p!7Oy+x.,mBx7");
      // Undeclared exception!
      try { 
        document0.child(1946);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 1946, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Document document0 = new Document("lfloor");
      Elements elements0 = document0.getElementsByAttributeValueContaining("lfloor", "lfloor");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Document document0 = new Document("textarea");
      Element element0 = document0.prependElement("textarea");
      Element element1 = element0.val("\n<textarea></textarea>");
      assertEquals("textarea", element1.nodeName());
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Document document0 = new Document("textarea");
      Element element0 = document0.prependElement("textarea");
      String string0 = element0.val();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Document document0 = new Document("98");
      Elements elements0 = document0.getElementsByAttributeValueNot("98", "98");
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Document document0 = new Document("Z&");
      Elements elements0 = document0.getElementsByAttribute("Z&");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Document document0 = new Document("180");
      // Undeclared exception!
      try { 
        document0.html("180");
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Document document0 = new Document("V");
      Elements elements0 = document0.getElementsByAttributeValue("V", "V");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Document document0 = new Document(" ");
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
  public void test14()  throws Throwable  {
      Document document0 = new Document("");
      // Undeclared exception!
      try { 
        document0.getElementsByAttributeValueStarting("", "");
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // The validated string is empty
         //
         verifyException("org.apache.commons.lang.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Tag tag0 = Tag.valueOf("wJ{O%(aN^");
      Attributes attributes0 = new Attributes();
      Element element0 = new Element(tag0, "qX", attributes0);
      String string0 = element0.nodeName();
      assertEquals("wj{o%(an^", string0);
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Document document0 = new Document("1-:r@ddlZM13j");
      Element element0 = document0.removeClass("1-:r@ddlZM13j");
      assertSame(document0, element0);
  }

  @Test(timeout = 4000)
  public void test17()  throws Throwable  {
      Document document0 = new Document("*5_g*mdi)z~mxvc+p6h");
      // Undeclared exception!
      try { 
        document0.select("*5_g*mdi)z~mxvc+p6h");
        fail("Expecting exception: IllegalStateException");
      
      } catch(IllegalStateException e) {
         //
         // Could not parse query *5_g*mdi)z~mxvc+p6h
         //
         verifyException("org.jsoup.select.Selector", e);
      }
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Document document0 = new Document("]W?lxF@m3E\"o/oA");
      Elements elements0 = document0.getElementsByIndexLessThan(4);
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Document document0 = new Document("||ysa6u*7f}x;");
      Elements elements0 = document0.getElementsByIndexGreaterThan(25);
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test20()  throws Throwable  {
      Document document0 = new Document("||ysa6u*7f}x;");
      Elements elements0 = document0.getElementsByIndexEquals(135);
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Document document0 = new Document("|,sapQ*J7f};");
      Element element0 = document0.getElementById("|,sapQ*J7f};");
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Document document0 = new Document("k");
      Element element0 = document0.prependElement("k");
      Elements elements0 = element0.parents();
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Tag tag0 = Tag.valueOf("* ");
      Element element0 = new Element(tag0, "* ");
      Element element1 = element0.prependElement("* ");
      Elements elements0 = element1.parents();
      assertFalse(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Document document0 = new Document("TBODY");
      Element element0 = document0.prependElement("TBODY");
      Element element1 = element0.previousElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Document document0 = new Document("4{p!7Oy+x.,mBx7");
      Element element0 = document0.prependElement("4{p!7Oy+x.,mBx7");
      Element element1 = element0.nextElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Document document0 = new Document("</");
      Element element0 = document0.prependElement("</");
      Element element1 = element0.firstElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Document document0 = new Document("P0r~#JG");
      document0.prependElement("P0r~#JG");
      document0.prependChild(document0);
      Element element0 = document0.firstElementSibling();
      assertSame(document0, element0);
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Document document0 = new Document("lf*oMr");
      Element element0 = document0.prependElement("lf*oMr");
      Integer integer0 = element0.elementSiblingIndex();
      assertEquals(0, (int)integer0);
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Tag tag0 = Tag.valueOf("* ");
      Element element0 = new Element(tag0, "* ");
      Element element1 = element0.prependElement("* ");
      Element element2 = element1.lastElementSibling();
      assertNull(element2);
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Document document0 = new Document("4{p!7Oy+x.,mx7");
      document0.appendChild(document0);
      Element element0 = document0.prependElement("4{p!7Oy+x.,mx7");
      Element element1 = element0.lastElementSibling();
      assertFalse(element1.isBlock());
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Document document0 = new Document("ieI}<r~");
      XmlDeclaration xmlDeclaration0 = new XmlDeclaration("ieI}<r~", "%kg[}j0l)%PY@;V2\"", false);
      document0.appendChild(xmlDeclaration0);
      String string0 = document0.text();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Tag tag0 = Tag.valueOf("* ");
      Element element0 = new Element(tag0, "* ");
      Element element1 = element0.prependText(";/@j!T!hDz%d'&i");
      element1.prependText("* ");
      String string0 = element1.text();
      assertEquals("* ;/@j!T!hDz%d'&i", string0);
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Document document0 = new Document("}");
      document0.prependElement("}");
      String string0 = document0.text();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Document document0 = new Document("}");
      document0.prependElement("}");
      document0.prependText("}");
      String string0 = document0.text();
      assertEquals("}", string0);
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Document document0 = new Document("textarea");
      Element element0 = document0.prependElement("textarea");
      boolean boolean0 = element0.preserveWhitespace();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Document document0 = new Document("lfloor");
      Element element0 = document0.prependElement("lfloor");
      boolean boolean0 = element0.preserveWhitespace();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Tag tag0 = Tag.valueOf("P6Jr~#JG");
      Element element0 = new Element(tag0, "Gk4U:qx@zY");
      Element element1 = element0.prependText("");
      boolean boolean0 = element1.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Tag tag0 = Tag.valueOf("* ");
      Element element0 = new Element(tag0, "* ");
      element0.prependElement("* ");
      boolean boolean0 = element0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Document document0 = new Document("ch^9$$!]9)x2bs`5ey");
      DataNode dataNode0 = new DataNode("<", "<");
      document0.appendChild(dataNode0);
      boolean boolean0 = document0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Document document0 = new Document("org.jsoup.nodes.Element");
      Element element0 = document0.prependElement("org.jsoup.nodes.Element");
      element0.prependText("org.jsoup.nodes.Element");
      boolean boolean0 = document0.hasText();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Document document0 = new Document("lf*oMr");
      document0.prependElement("lf*oMr");
      String string0 = document0.data();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Document document0 = new Document("]W?lxF@m3E\"o/oA");
      document0.appendText("]W?lxF@m3E\"o/oA");
      String string0 = document0.data();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Document document0 = new Document("s+");
      LinkedHashSet<String> linkedHashSet0 = new LinkedHashSet<String>();
      Element element0 = document0.classNames((Set<String>) linkedHashSet0);
      Element element1 = element0.toggleClass("s+");
      assertSame(element0, element1);
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Document document0 = new Document("8");
      Element element0 = document0.toggleClass("8");
      Element element1 = element0.addClass("8");
      assertEquals("8", element1.baseUri());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Document document0 = new Document("4{p!7Oy+x.,mBx7");
      boolean boolean0 = document0.hasClass("");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test46()  throws Throwable  {
      Document document0 = new Document("]W?lxF@m3E\"o/oA");
      Element element0 = document0.toggleClass("");
      assertEquals("#document", element0.nodeName());
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Tag tag0 = Tag.valueOf("* ");
      Element element0 = new Element(tag0, "* ");
      String string0 = element0.val();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Document document0 = new Document("TBODY");
      Element element0 = document0.prependElement("TBODY");
      String string0 = element0.toString();
      assertEquals("\n<tbody>\n</tbody>", string0);
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Document document0 = new Document("</");
      document0.prependElement("</");
      Element element0 = document0.prependText("</");
      String string0 = element0.toString();
      assertEquals("&lt;/<</>\n</</>", string0);
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Tag tag0 = Tag.valueOf("*k");
      Element element0 = new Element(tag0, "*k");
      Element element1 = element0.prependElement("*k");
      String string0 = element0.toString();
      assertFalse(element0.equals((Object)element1));
      assertEquals("<*k>\n<*k>\n</*k>\n</*k>", string0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Document document0 = new Document("4{p!7Oy+x.,mBx7");
      Element element0 = document0.prependElement("4{p!7Oy+x.,mBx7");
      boolean boolean0 = document0.equals(element0);
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Document document0 = new Document("4{p!7Oy+x.,mBx7");
      boolean boolean0 = document0.equals("4{p!7Oy+x.,mBx7");
      assertFalse(boolean0);
  }
}