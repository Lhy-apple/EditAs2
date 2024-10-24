/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:07:06 GMT 2023
 */

package org.jsoup.nodes;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.*;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.regex.Pattern;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.jsoup.nodes.Attributes;
import org.jsoup.nodes.Comment;
import org.jsoup.nodes.DataNode;
import org.jsoup.nodes.Document;
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
      Document document0 = new Document(";");
      Elements elements0 = document0.getElementsMatchingText(";");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test01()  throws Throwable  {
      Tag tag0 = Tag.valueOf("br");
      Element element0 = new Element(tag0, "br");
      element0.prependChild(element0);
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
  public void test02()  throws Throwable  {
      Tag tag0 = Tag.valueOf("br");
      Element element0 = new Element(tag0, "br");
      Element element1 = element0.appendElement("del");
      assertEquals("del", element1.tagName());
      
      element0.text();
      assertNotSame(element0, element1);
  }

  @Test(timeout = 4000)
  public void test03()  throws Throwable  {
      Document document0 = new Document("z=P|6]nFsYvM&EW:");
      // Undeclared exception!
      try { 
        document0.child(771);
        fail("Expecting exception: IndexOutOfBoundsException");
      
      } catch(IndexOutOfBoundsException e) {
         //
         // Index: 771, Size: 0
         //
         verifyException("java.util.ArrayList", e);
      }
  }

  @Test(timeout = 4000)
  public void test04()  throws Throwable  {
      Tag tag0 = Tag.valueOf("br");
      Element element0 = new Element(tag0, "br");
      Element element1 = element0.appendText("br");
      String string0 = element1.outerHtml();
      assertEquals("<br>br</br>", string0);
  }

  @Test(timeout = 4000)
  public void test05()  throws Throwable  {
      Document document0 = new Document("(Q9q%XX(");
      Element element0 = document0.prependText("(Q9q%XX(");
      assertEquals("#root", element0.tagName());
  }

  @Test(timeout = 4000)
  public void test06()  throws Throwable  {
      Document document0 = new Document("noQuirks");
      Document document1 = (Document)document0.tagName("noQuirks");
      assertEquals(Document.QuirksMode.noQuirks, document1.quirksMode());
  }

  @Test(timeout = 4000)
  public void test07()  throws Throwable  {
      Document document0 = new Document("jm=#5");
      // Undeclared exception!
      try { 
        document0.html("jm=#5");
        fail("Expecting exception: NoSuchElementException");
      
      } catch(NoSuchElementException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("java.util.LinkedList", e);
      }
  }

  @Test(timeout = 4000)
  public void test08()  throws Throwable  {
      Document document0 = new Document(",G'+VsH\"av;2Q9[");
      Map<String, String> map0 = document0.dataset();
      assertEquals(0, map0.size());
  }

  @Test(timeout = 4000)
  public void test09()  throws Throwable  {
      Document document0 = new Document("InCell");
      Elements elements0 = document0.getElementsContainingOwnText("InCell");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test10()  throws Throwable  {
      Document document0 = new Document("noQirks");
      Element element0 = document0.addClass("noQirks");
      assertSame(document0, element0);
  }

  @Test(timeout = 4000)
  public void test11()  throws Throwable  {
      Document document0 = new Document("Ta%0[z97b44");
      Elements elements0 = document0.getElementsByAttributeValue("Ta%0[z97b44", "Ta%0[z97b44");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test12()  throws Throwable  {
      Document document0 = new Document("Ta%0[z97b44");
      Elements elements0 = document0.getElementsByAttributeValueStarting("Ta%0[z97b44", "Ta%0[z97b44");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test13()  throws Throwable  {
      Document document0 = new Document("noQuirks");
      Elements elements0 = document0.select("noQuirks");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test14()  throws Throwable  {
      Tag tag0 = Tag.valueOf("br");
      Element element0 = new Element(tag0, "br");
      Element element1 = element0.prepend("br");
      assertFalse(element1.isBlock());
  }

  @Test(timeout = 4000)
  public void test15()  throws Throwable  {
      Document document0 = new Document("InCell");
      Elements elements0 = document0.getElementsMatchingOwnText("InCell");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test16()  throws Throwable  {
      Document document0 = new Document("WN`+,2^K(z?#");
      // Undeclared exception!
      try { 
        document0.after("6g`\"E^,");
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
      Document document0 = new Document(",'+Vs\"a;2Q9[");
      Elements elements0 = document0.getElementsByIndexLessThan((-1934726696));
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test18()  throws Throwable  {
      Document document0 = new Document("Ta%0[z97b44");
      Elements elements0 = document0.parents();
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test19()  throws Throwable  {
      Document document0 = new Document("ScriptDataDoubleEscapedDashDash");
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
  public void test20()  throws Throwable  {
      Document document0 = new Document("WN`+,2^K(z?#");
      Elements elements0 = document0.getElementsByAttributeStarting("WN`+,2^K(z?#");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test21()  throws Throwable  {
      Document document0 = new Document("noQuik");
      Elements elements0 = document0.getElementsByIndexEquals((-4137));
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test22()  throws Throwable  {
      Document document0 = new Document("_Vsa39N)");
      // Undeclared exception!
      try { 
        document0.wrap("#root");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Node", e);
      }
  }

  @Test(timeout = 4000)
  public void test23()  throws Throwable  {
      Document document0 = new Document("B`xf");
      Element element0 = document0.val("B`xf");
      assertEquals(0, element0.siblingIndex());
  }

  @Test(timeout = 4000)
  public void test24()  throws Throwable  {
      Document document0 = new Document("[t$7|2n");
      // Undeclared exception!
      try { 
        document0.getElementsByAttributeValueMatching("G\"Iu;.VCIN$#V", (String) null);
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test25()  throws Throwable  {
      Document document0 = new Document("noQuirks");
      Elements elements0 = document0.getElementsByAttributeValueEnding("vAdxzF;';Jq<(", "vAdxzF;';Jq<(");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test26()  throws Throwable  {
      Document document0 = new Document("org.jsoup.select.Evaluator$AttributeKeyPair");
      Elements elements0 = document0.getElementsByClass("org.jsoup.select.Evaluator$AttributeKeyPair");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test27()  throws Throwable  {
      Document document0 = new Document("Insert position out of bounds.");
      Elements elements0 = document0.getElementsContainingText("Insert position out of bounds.");
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test28()  throws Throwable  {
      Document document0 = new Document("B`xf");
      Pattern pattern0 = Pattern.compile("B`xf");
      Elements elements0 = document0.getElementsByAttributeValueMatching("B`xf", pattern0);
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test29()  throws Throwable  {
      Document document0 = new Document("InCell");
      Elements elements0 = document0.getElementsByAttributeValueContaining("InCell", "InCell");
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test30()  throws Throwable  {
      Document document0 = new Document(",'+Vs\"a;2Q9[");
      Element element0 = document0.appendElement(",'+Vs\"a;2Q9[");
      element0.text(",'+Vs\"a;2Q9[");
      String string0 = element0.ownText();
      assertEquals(",'+Vs\"a;2Q9[", string0);
  }

  @Test(timeout = 4000)
  public void test31()  throws Throwable  {
      Document document0 = new Document("BeforeDoctypePublicIdentifier");
      // Undeclared exception!
      try { 
        document0.getElementsByAttributeValueNot((String) null, (String) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // String must not be empty
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test32()  throws Throwable  {
      Document document0 = new Document("noQuik");
      Elements elements0 = document0.getElementsByAttribute("noQuik");
      Element element0 = document0.insertChildren((-1), elements0);
      assertEquals("noQuik", element0.baseUri());
  }

  @Test(timeout = 4000)
  public void test33()  throws Throwable  {
      Document document0 = new Document("s5t");
      // Undeclared exception!
      try { 
        document0.before((String) null);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Object must not be null
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test34()  throws Throwable  {
      Document document0 = new Document("ucjb!;");
      Elements elements0 = document0.getAllElements();
      assertFalse(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test35()  throws Throwable  {
      Document document0 = new Document("9wb)I`n 1_:H");
      // Undeclared exception!
      try { 
        document0.title("9wb)I`n 1_:H");
        fail("Expecting exception: NullPointerException");
      
      } catch(NullPointerException e) {
         //
         // no message in exception (getMessage() returned null)
         //
         verifyException("org.jsoup.nodes.Document", e);
      }
  }

  @Test(timeout = 4000)
  public void test36()  throws Throwable  {
      Document document0 = new Document("(Q9q%XX(");
      Elements elements0 = document0.getElementsByIndexGreaterThan(31);
      assertEquals(0, elements0.size());
  }

  @Test(timeout = 4000)
  public void test37()  throws Throwable  {
      Tag tag0 = Tag.valueOf("br");
      Element element0 = new Element(tag0, "br");
      Element element1 = element0.getElementById("ruby");
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test38()  throws Throwable  {
      Document document0 = new Document("Ta%0[z97b44");
      Element element0 = document0.appendElement("Ta%0[z97b44");
      Elements elements0 = element0.parents();
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test39()  throws Throwable  {
      Tag tag0 = Tag.valueOf("br");
      Attributes attributes0 = new Attributes();
      Element element0 = new Element(tag0, "br", attributes0);
      Element element1 = element0.prependChild(element0);
      element1.parents();
  }

  @Test(timeout = 4000)
  public void test40()  throws Throwable  {
      Tag tag0 = Tag.valueOf("Xg2V4Wy");
      Element element0 = new Element(tag0, "Xg2V4Wy");
      element0.prependChild(element0);
      element0.appendText("Xg2V4Wy");
      Element element1 = element0.lastElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test41()  throws Throwable  {
      Tag tag0 = Tag.valueOf("PLAINTEXT");
      Element element0 = new Element(tag0, "s+");
      Element element1 = element0.text("Xg4Wy");
      List<TextNode> list0 = element1.textNodes();
      assertEquals(1, list0.size());
  }

  @Test(timeout = 4000)
  public void test42()  throws Throwable  {
      Tag tag0 = Tag.valueOf("Xg2V4Wy");
      Element element0 = new Element(tag0, "Xg2V4Wy");
      element0.appendElement("Xg2V4Wy");
      List<TextNode> list0 = element0.textNodes();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test43()  throws Throwable  {
      Document document0 = new Document("Ta%0[z9U]67b44");
      DataNode dataNode0 = new DataNode("class", "I+\"%fWbqP8.-");
      Element element0 = document0.prependChild(dataNode0);
      List<DataNode> list0 = element0.dataNodes();
      assertTrue(list0.contains(dataNode0));
  }

  @Test(timeout = 4000)
  public void test44()  throws Throwable  {
      Document document0 = new Document("<!");
      document0.appendElement("<!");
      List<DataNode> list0 = document0.dataNodes();
      assertTrue(list0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test45()  throws Throwable  {
      Document document0 = new Document(",'+Vs\"a;2Q9[");
      List<DataNode> list0 = document0.dataNodes();
      // Undeclared exception!
      try { 
        document0.insertChildren(90, list0);
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
      Document document0 = new Document("menu");
      List<DataNode> list0 = document0.dataNodes();
      // Undeclared exception!
      try { 
        document0.insertChildren((-2923), list0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // Insert position out of bounds.
         //
         verifyException("org.jsoup.helper.Validate", e);
      }
  }

  @Test(timeout = 4000)
  public void test47()  throws Throwable  {
      Document document0 = new Document("_/Vsa39N)8");
      Elements elements0 = document0.siblingElements();
      assertTrue(elements0.isEmpty());
  }

  @Test(timeout = 4000)
  public void test48()  throws Throwable  {
      Tag tag0 = Tag.valueOf("Xg4Wy");
      Element element0 = new Element(tag0, "Xg4Wy");
      Element element1 = element0.appendElement("cg~");
      Element element2 = element1.after((Node) element0);
      assertEquals("cg~", element2.nodeName());
      
      Elements elements0 = element0.siblingElements();
      assertEquals(1, elements0.size());
  }

  @Test(timeout = 4000)
  public void test49()  throws Throwable  {
      Tag tag0 = Tag.valueOf("Xg4Wy");
      Element element0 = new Element(tag0, "Xg4Wy");
      Element element1 = element0.appendElement("cg~");
      assertFalse(element1.equals((Object)element0));
      
      Element element2 = element1.nextElementSibling();
      assertNull(element2);
      assertEquals("cg~", element1.tagName());
  }

  @Test(timeout = 4000)
  public void test50()  throws Throwable  {
      Document document0 = new Document(" />");
      Element element0 = document0.nextElementSibling();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test51()  throws Throwable  {
      Tag tag0 = Tag.valueOf("Xg4Wy");
      Element element0 = new Element(tag0, "Xg4Wy");
      Element element1 = element0.appendElement("cg~");
      element1.after((Node) element0);
      Element element2 = element1.nextElementSibling();
      assertNotNull(element2);
      assertEquals("xg4wy", element2.nodeName());
      assertEquals("cg~", element1.tagName());
  }

  @Test(timeout = 4000)
  public void test52()  throws Throwable  {
      Document document0 = new Document("<!");
      Element element0 = document0.appendElement("<!");
      Element element1 = element0.previousElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test53()  throws Throwable  {
      Document document0 = new Document("InCell");
      Element element0 = document0.previousElementSibling();
      assertNull(element0);
  }

  @Test(timeout = 4000)
  public void test54()  throws Throwable  {
      Document document0 = new Document("x2{.b");
      Element element0 = document0.appendElement("x2{.b");
      element0.after((Node) document0);
      Element element1 = document0.previousElementSibling();
      assertEquals("x2{.b", element1.nodeName());
  }

  @Test(timeout = 4000)
  public void test55()  throws Throwable  {
      Document document0 = new Document("ScriptDataDoubleEscapedDashDash");
      Element element0 = document0.appendElement("ScriptDataDoubleEscapedDashDash");
      Element element1 = element0.firstElementSibling();
      assertNull(element1);
  }

  @Test(timeout = 4000)
  public void test56()  throws Throwable  {
      Tag tag0 = Tag.valueOf("x2{.b");
      Element element0 = new Element(tag0, "x2{.b");
      Element element1 = element0.appendElement("br");
      Element element2 = element1.before((Node) element0);
      assertEquals("br", element2.nodeName());
      
      Element element3 = element0.firstElementSibling();
      assertNotNull(element3);
      assertEquals("x2{.b", element3.tagName());
  }

  @Test(timeout = 4000)
  public void test57()  throws Throwable  {
      Document document0 = new Document("ScriptDataDoubleEscapedDashDash");
      Element element0 = document0.appendElement("ScriptDataDoubleEscapedDashDash");
      Integer integer0 = element0.elementSiblingIndex();
      assertEquals(0, (int)integer0);
  }

  @Test(timeout = 4000)
  public void test58()  throws Throwable  {
      Document document0 = new Document("Ta%0[z97b44");
      Element element0 = document0.appendElement("br");
      Element element1 = element0.after((Node) document0);
      Element element2 = element1.lastElementSibling();
      assertNotNull(element2);
      assertEquals("br", element1.tagName());
  }

  @Test(timeout = 4000)
  public void test59()  throws Throwable  {
      Document document0 = new Document("InCell");
      document0.setParentNode(document0);
      Integer integer0 = document0.elementSiblingIndex();
      assertNull(integer0);
  }

  @Test(timeout = 4000)
  public void test60()  throws Throwable  {
      Document document0 = new Document("B`xf");
      Comment comment0 = new Comment(";", "1j'n>h*l+t(!)p /t");
      document0.prependChild(comment0);
      String string0 = document0.text();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test61()  throws Throwable  {
      Document document0 = new Document("noQuik");
      document0.prependChild(document0);
      // Undeclared exception!
      try { 
        document0.text();
        fail("Expecting exception: StackOverflowError");
      
      } catch(StackOverflowError e) {
         //
         // no message in exception (getMessage() returned null)
         //
      }
  }

  @Test(timeout = 4000)
  public void test62()  throws Throwable  {
      Tag tag0 = Tag.valueOf("br");
      Element element0 = new Element(tag0, "br");
      element0.appendText("br");
      Element element1 = element0.appendElement("del");
      assertEquals("del", element1.nodeName());
      
      String string0 = element0.text();
      assertEquals("br", string0);
  }

  @Test(timeout = 4000)
  public void test63()  throws Throwable  {
      Tag tag0 = Tag.valueOf("br");
      Element element0 = new Element(tag0, "br");
      element0.appendElement("del");
      String string0 = element0.ownText();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test64()  throws Throwable  {
      Document document0 = new Document("B`xf");
      Comment comment0 = new Comment(";", "1j'n>h*l+t(!)p /t");
      document0.prependChild(comment0);
      String string0 = document0.ownText();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test65()  throws Throwable  {
      Tag tag0 = Tag.valueOf("PLAINTEXT");
      Element element0 = new Element(tag0, "s+");
      Element element1 = element0.text("Xg4Wy");
      String string0 = element1.ownText();
      assertEquals("Xg4Wy", string0);
  }

  @Test(timeout = 4000)
  public void test66()  throws Throwable  {
      Tag tag0 = Tag.valueOf("textarea");
      Attributes attributes0 = new Attributes();
      Element element0 = new Element(tag0, "_Vsa39N)8", attributes0);
      Element element1 = element0.appendElement("$Z");
      boolean boolean0 = element1.preserveWhitespace();
      assertTrue(boolean0);
      assertEquals("$z", element1.nodeName());
  }

  @Test(timeout = 4000)
  public void test67()  throws Throwable  {
      Document document0 = new Document("RcdataLessthanSign");
      document0.appendElement("RcdataLessthanSign");
      boolean boolean0 = document0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test68()  throws Throwable  {
      Document document0 = new Document("WN`+,2^K(z?#");
      document0.append(" ");
      boolean boolean0 = document0.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test69()  throws Throwable  {
      Tag tag0 = Tag.valueOf("s+");
      Element element0 = new Element(tag0, "s+");
      DataNode dataNode0 = DataNode.createFromEncoded("#root", "s+");
      Element element1 = element0.prependChild(dataNode0);
      boolean boolean0 = element1.hasText();
      assertFalse(boolean0);
  }

  @Test(timeout = 4000)
  public void test70()  throws Throwable  {
      Document document0 = new Document("_/Vsa39N)8");
      Document document1 = document0.clone();
      document1.append("ZU6nXt}?d62X@`;jBv");
      document0.prependChild(document1);
      assertEquals(1, document1.childNodeSize());
      
      boolean boolean0 = document0.hasText();
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test71()  throws Throwable  {
      Document document0 = new Document("Ta%0[z9U]67b44");
      DataNode dataNode0 = new DataNode("class", "I+\"%fWbqP8.-");
      Element element0 = document0.prependChild(dataNode0);
      String string0 = element0.data();
      assertEquals("class", string0);
  }

  @Test(timeout = 4000)
  public void test72()  throws Throwable  {
      Document document0 = new Document("Ta%0[z97b44");
      Comment comment0 = new Comment("Ta%0[z97b44", "Ta%0[z97b44");
      Element element0 = document0.prependChild(comment0);
      String string0 = element0.data();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test73()  throws Throwable  {
      Document document0 = new Document("ScriptDataDoubleEscapedDashDash");
      document0.appendElement("ScriptDataDoubleEscapedDashDash");
      String string0 = document0.data();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test74()  throws Throwable  {
      Document document0 = new Document(",'+Vs\"a;2Q9[");
      document0.removeClass(",'+Vs\"a;2Q9[");
      Element element0 = document0.clone();
      assertEquals(0, element0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test75()  throws Throwable  {
      Document document0 = new Document("fwVL9D([No,c/ou9Z");
      boolean boolean0 = document0.hasClass("");
      assertTrue(boolean0);
  }

  @Test(timeout = 4000)
  public void test76()  throws Throwable  {
      Document document0 = new Document(",'+Vs\"a;2Q9[");
      Element element0 = document0.toggleClass(",'+Vs\"a;2Q9[");
      assertEquals(0, element0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test77()  throws Throwable  {
      Document document0 = new Document("m48eAtCr}%5=H");
      Element element0 = document0.toggleClass("");
      assertSame(document0, element0);
  }

  @Test(timeout = 4000)
  public void test78()  throws Throwable  {
      Tag tag0 = Tag.valueOf("br");
      Element element0 = new Element(tag0, "br");
      String string0 = element0.val();
      assertEquals("", string0);
  }

  @Test(timeout = 4000)
  public void test79()  throws Throwable  {
      Document document0 = new Document("VmPJVA-ZrN-");
      Element element0 = document0.appendElement("textarea");
      element0.val();
      assertEquals("textarea", element0.tagName());
  }

  @Test(timeout = 4000)
  public void test80()  throws Throwable  {
      Tag tag0 = Tag.valueOf("textarea");
      Element element0 = new Element(tag0, "y-!");
      element0.val("y-!");
      assertEquals(1, element0.childNodeSize());
  }

  @Test(timeout = 4000)
  public void test81()  throws Throwable  {
      Document document0 = new Document(",'+Vs\"a;2Q9[");
      StringBuilder stringBuilder0 = new StringBuilder(",'+Vs\"a;2Q9[");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.prettyPrint(false);
      document0.outerHtmlHead(stringBuilder0, (-131938718), document_OutputSettings1);
      assertEquals(",'+Vs\"a;2Q9[<#root>", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test82()  throws Throwable  {
      Document document0 = new Document("#<Gc");
      Element element0 = document0.appendElement("br");
      StringBuilder stringBuilder0 = new StringBuilder("l");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      // Undeclared exception!
      try { 
        element0.outerHtmlHead(stringBuilder0, (-1911450750), document_OutputSettings0);
        fail("Expecting exception: IllegalArgumentException");
      
      } catch(IllegalArgumentException e) {
         //
         // width must be > 0
         //
         verifyException("org.jsoup.helper.StringUtil", e);
      }
  }

  @Test(timeout = 4000)
  public void test83()  throws Throwable  {
      Tag tag0 = Tag.valueOf("br");
      Element element0 = new Element(tag0, "br");
      StringBuilder stringBuilder0 = new StringBuilder("?d,)moy7a");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      element0.outerHtmlHead(stringBuilder0, 1201, document_OutputSettings0);
      assertEquals("?d,)moy7a<br />", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test84()  throws Throwable  {
      Tag tag0 = Tag.valueOf("br");
      Element element0 = new Element(tag0, "#<Gc");
      Element element1 = element0.appendElement("br");
      StringBuilder stringBuilder0 = new StringBuilder("li");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      element1.outerHtmlHead(stringBuilder0, (-1201601922), document_OutputSettings0);
      assertEquals("li<br />", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test85()  throws Throwable  {
      Tag tag0 = Tag.valueOf("br");
      Element element0 = new Element(tag0, "br");
      String string0 = element0.outerHtml();
      assertEquals("<br />", string0);
  }

  @Test(timeout = 4000)
  public void test86()  throws Throwable  {
      Document document0 = new Document(",'+Vs\"a;2Q9[");
      StringBuilder stringBuilder0 = new StringBuilder(",'+Vs\"a;2Q9[");
      Document.OutputSettings document_OutputSettings0 = new Document.OutputSettings();
      Document.OutputSettings document_OutputSettings1 = document_OutputSettings0.prettyPrint(false);
      document0.outerHtmlTail(stringBuilder0, (-2623), document_OutputSettings1);
      assertEquals(",'+Vs\"a;2Q9[</#root>", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test87()  throws Throwable  {
      Document document0 = new Document(",G'+VsH\"av;2Q9[");
      document0.appendElement(",G'+VsH\"av;2Q9[");
      StringBuilder stringBuilder0 = new StringBuilder("<");
      document0.outerHtml(stringBuilder0);
      assertEquals("<\n<#root>\n <,g'+vsh\"av;2q9[></,g'+vsh\"av;2q9[>\n</#root>", stringBuilder0.toString());
  }

  @Test(timeout = 4000)
  public void test88()  throws Throwable  {
      Document document0 = new Document("InCell");
      document0.append("InCell");
      String string0 = document0.toString();
      assertEquals("InCell", string0);
  }

  @Test(timeout = 4000)
  public void test89()  throws Throwable  {
      Tag tag0 = Tag.valueOf("br");
      Element element0 = new Element(tag0, "br");
      element0.hashCode();
  }
}
