/*
 * This file was automatically generated by EvoSuite
 * Wed Jul 12 06:18:56 GMT 2023
 */

package org.apache.commons.jxpath.ri.model.dom;

import org.junit.Test;
import static org.junit.Assert.*;
import java.util.Locale;
import org.apache.commons.jxpath.JXPathContext;
import org.apache.commons.jxpath.ri.QName;
import org.apache.commons.jxpath.ri.model.NodePointer;
import org.apache.commons.jxpath.ri.model.dom.DOMAttributeIterator;
import org.apache.html.dom.HTMLDListElementImpl;
import org.apache.html.dom.HTMLDocumentImpl;
import org.apache.html.dom.HTMLHtmlElementImpl;
import org.apache.html.dom.HTMLTextAreaElementImpl;
import org.apache.wml.dom.WMLDocumentImpl;
import org.apache.wml.dom.WMLTimerElementImpl;
import org.apache.xerces.dom.DocumentTypeImpl;
import org.evosuite.runtime.EvoRunner;
import org.evosuite.runtime.EvoRunnerParameters;
import org.junit.runner.RunWith;

@RunWith(EvoRunner.class) @EvoRunnerParameters(mockJVMNonDeterminism = true, useVFS = true, useVNET = true, resetStaticState = true, separateClassLoader = true) 
public class DOMAttributeIterator_ESTest extends DOMAttributeIterator_ESTest_scaffolding {

  @Test(timeout = 4000)
  public void test0()  throws Throwable  {
      QName qName0 = new QName("");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLDListElementImpl hTMLDListElementImpl0 = new HTMLDListElementImpl(hTMLDocumentImpl0, "");
      Locale locale0 = Locale.TAIWAN;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLDListElementImpl0, locale0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      int int0 = dOMAttributeIterator0.getPosition();
      assertEquals(0, int0);
  }

  @Test(timeout = 4000)
  public void test1()  throws Throwable  {
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      Locale locale0 = new Locale("xmlns", "xmlns");
      QName qName0 = new QName("xmlns", "xmlns");
      HTMLHtmlElementImpl hTMLHtmlElementImpl0 = new HTMLHtmlElementImpl(hTMLDocumentImpl0, "eM{:");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLHtmlElementImpl0, locale0);
      NodePointer nodePointer1 = nodePointer0.createAttribute((JXPathContext) null, qName0);
      assertEquals(1, nodePointer1.getLength());
  }

  @Test(timeout = 4000)
  public void test2()  throws Throwable  {
      QName qName0 = new QName("*", "*");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLDListElementImpl hTMLDListElementImpl0 = new HTMLDListElementImpl(hTMLDocumentImpl0, "*");
      Locale locale0 = Locale.PRC;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLDListElementImpl0, locale0);
      hTMLDListElementImpl0.setLang("*");
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      NodePointer nodePointer1 = dOMAttributeIterator0.getNodePointer();
      assertNotNull(nodePointer1);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test3()  throws Throwable  {
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      QName qName0 = new QName("xmlns", "xmlns");
      Locale locale0 = Locale.CHINA;
      HTMLTextAreaElementImpl hTMLTextAreaElementImpl0 = new HTMLTextAreaElementImpl(hTMLDocumentImpl0, "xmlns");
      hTMLTextAreaElementImpl0.setAttribute("xmlns", "xmlns");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLTextAreaElementImpl0, locale0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test4()  throws Throwable  {
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLDListElementImpl hTMLDListElementImpl0 = new HTMLDListElementImpl(hTMLDocumentImpl0, "xmlns");
      hTMLDListElementImpl0.setLang("xmlns");
      Locale locale0 = new Locale("xmlns", "xmlns");
      QName qName0 = new QName("xmlns", "xmlns");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLDListElementImpl0, locale0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test5()  throws Throwable  {
      QName qName0 = new QName("*");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLDListElementImpl hTMLDListElementImpl0 = new HTMLDListElementImpl(hTMLDocumentImpl0, "*");
      Locale locale0 = Locale.US;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLDListElementImpl0, locale0);
      hTMLDListElementImpl0.setLang("*");
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test6()  throws Throwable  {
      QName qName0 = new QName("*");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      Locale locale0 = new Locale("*", "*");
      DocumentTypeImpl documentTypeImpl0 = new DocumentTypeImpl(hTMLDocumentImpl0, "*");
      WMLDocumentImpl wMLDocumentImpl0 = new WMLDocumentImpl(documentTypeImpl0);
      WMLTimerElementImpl wMLTimerElementImpl0 = new WMLTimerElementImpl(wMLDocumentImpl0, "*");
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, wMLTimerElementImpl0, locale0);
      wMLTimerElementImpl0.setXmlLang("*");
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test7()  throws Throwable  {
      QName qName0 = new QName("*", "*");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      HTMLDListElementImpl hTMLDListElementImpl0 = new HTMLDListElementImpl(hTMLDocumentImpl0, "*");
      Locale locale0 = Locale.PRC;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLDListElementImpl0, locale0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      assertEquals(0, dOMAttributeIterator0.getPosition());
      
      dOMAttributeIterator0.getNodePointer();
      assertEquals(1, dOMAttributeIterator0.getPosition());
  }

  @Test(timeout = 4000)
  public void test8()  throws Throwable  {
      QName qName0 = new QName("Aw'ICI-SIJ`j.Q]}{~e");
      HTMLDocumentImpl hTMLDocumentImpl0 = new HTMLDocumentImpl();
      Locale locale0 = Locale.CANADA;
      NodePointer nodePointer0 = NodePointer.newNodePointer(qName0, hTMLDocumentImpl0, locale0);
      DOMAttributeIterator dOMAttributeIterator0 = new DOMAttributeIterator(nodePointer0, qName0);
      boolean boolean0 = dOMAttributeIterator0.setPosition((-2673));
      assertEquals((-2673), dOMAttributeIterator0.getPosition());
      assertFalse(boolean0);
  }
}
